import os
import boto3
import tempfile
import time
import logging
from typing import List, Tuple, Optional
import polars as pl
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('conte-transform')


# Include the processing functions from the original code
def standardize_job_id(id_series: pl.Series) -> pl.Series:
    """Convert jobIDxxxxx to JOBxxxxx"""
    return (
        pl.when(id_series.str.contains('^jobID'))
        .then(pl.concat_str([pl.lit('JOB'), id_series.str.slice(5)]))
        .otherwise(id_series)
    )


def process_chunk(jobs_df: pl.DataFrame, ts_chunk: pl.DataFrame) -> pl.DataFrame:
    """Process a chunk of time series data against all jobs"""
    # Join time series chunk with jobs data on jobID
    joined = ts_chunk.join(
        jobs_df,
        left_on="Job Id",
        right_on="jobID",
        how="inner"
    )

    # Filter timestamps that fall between job start and end times
    filtered = joined.filter(
        (pl.col("Timestamp") >= pl.col("start"))
        & (pl.col("Timestamp") <= pl.col("end"))
    )

    if filtered.height == 0:
        return None

    # Pivot the metrics into columns
    # Group by all columns except Event and Value
    group_cols = [col for col in filtered.columns if col not in ["Event", "Value"]]

    result = filtered.pivot(
        values="Value",
        index=group_cols,
        on="Event",
        aggregate_function="first"
    )

    # Rename metric columns to the desired format
    for col in result.columns:
        if col in ["cpuuser", "gpu", "memused", "memused_minus_diskcache", "nfs", "block"]:
            result = result.rename({col: f"value_{col}"})

    return result


def join_job_timeseries(job_file: str, timeseries_file: str, output_file: str, chunk_size: int = 100_000):
    """
    Join job accounting data with time series data, creating a row for each timestamp.
    """
    # Define datetime formats for both data sources
    JOB_DATETIME_FMT = "%m/%d/%Y %H:%M:%S"  # Format for job data: "03/01/2015 01:29:34"
    TS_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"  # Format for timeseries data: "2015-03-01 14:56:51"

    logger.info("Reading job accounting data...")

    # Read jobs data with schema overrides for columns that have mixed types
    jobs_df = pl.scan_csv(
        job_file,
        schema_overrides={
            "Resource_List.neednodes": pl.Utf8,
            "Resource_List.nodes": pl.Utf8,
        }
    ).collect()

    # Standardize job IDs
    jobs_df = jobs_df.with_columns([
        standardize_job_id(pl.col("jobID")).alias("jobID")
    ])

    # Convert the start and end columns to datetime using the job data format
    jobs_df = jobs_df.with_columns([
        pl.col("start").str.strptime(pl.Datetime, JOB_DATETIME_FMT).alias("start"),
        pl.col("end").str.strptime(pl.Datetime, JOB_DATETIME_FMT).alias("end")
    ])

    logger.info("Processing time series data in chunks...")

    # Read the time series data
    ts_reader = pl.scan_csv(timeseries_file).collect()
    total_rows = ts_reader.height
    chunks = range(0, total_rows, chunk_size)
    first_chunk = True

    for chunk_start in tqdm(chunks):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        ts_chunk = ts_reader[chunk_start:chunk_end]

        # Convert the Timestamp column to datetime using the timeseries format
        ts_chunk = ts_chunk.with_columns([
            pl.col("Timestamp").str.strptime(pl.Datetime, TS_DATETIME_FMT)
        ])

        # Process the chunk by joining and filtering
        result_df = process_chunk(jobs_df, ts_chunk)

        if result_df is not None:
            # Write results to the output CSV file
            if first_chunk:
                # Write with headers for first chunk
                result_df.write_csv(output_file, include_header=True)
                first_chunk = False
            else:
                # For subsequent chunks, append by writing to a temporary file and concatenating
                temp_file = output_file + '.tmp'
                result_df.write_csv(temp_file, include_header=False)

                # Read the temporary file content
                with open(temp_file, 'r', encoding='utf-8') as temp:
                    content = temp.read()

                # Append to the main file
                with open(output_file, 'a', encoding='utf-8') as main:
                    main.write(content)

                # Clean up temporary file
                import os
                os.remove(temp_file)


def get_s3_client():
    """Create and return an S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )


def list_s3_files(bucket_name: str, prefix: str = "") -> List[str]:
    """List all files in an S3 bucket with an optional prefix"""
    s3_client = get_s3_client()

    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    files = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                files.append(obj['Key'])

    return files


def download_s3_file(bucket_name: str, s3_key: str, local_path: str) -> bool:
    """Download a file from S3 to a local path"""
    try:
        s3_client = get_s3_client()

        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file
        logger.info(f"Downloading {s3_key} to {local_path}")
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except Exception as e:
        logger.error(f"Error downloading {s3_key}: {str(e)}")
        return False


def upload_to_s3(file_paths: List[str], bucket_name="conte-transformed") -> bool:
    """Upload files to S3 public bucket without requiring credentials"""
    logger.info("\nStarting S3 upload...")

    s3_client = get_s3_client()

    for i, file_path in enumerate(file_paths, 1):
        file_name = os.path.basename(file_path)
        for attempt in range(3):
            try:
                # Add content type for CSV files
                extra_args = {
                    'ContentType': 'text/csv'
                }

                logger.info(f"Uploading {file_path} to {bucket_name}/{file_name}")
                s3_client.upload_file(
                    file_path,
                    bucket_name,
                    file_name,
                    ExtraArgs=extra_args
                )

                if i % 2 == 0 or i == len(file_paths):
                    logger.info(f"Uploaded {i}/{len(file_paths)} files")
                break
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to upload {file_name}: {str(e)}")
                    return False
                time.sleep(2 ** attempt)

    return True


def parse_time_series_filename(filename: str) -> Optional[Tuple[int, int]]:
    """Parse year and month from a timeseries filename"""
    pattern = r'FRESCO_Conte_ts_(\d{4})_(\d{2})_v\d+\.csv'
    match = re.match(pattern, os.path.basename(filename))
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return (year, month)
    return None


def get_job_filename_for_period(job_files: List[str], year: int, month: int) -> Optional[str]:
    """Find the job file for a specific year and month"""
    # Job files are named by year-month combination
    year_str = str(year)
    month_str = f"{month:02d}"

    # First try to find an exact match with year-month pattern
    for file in job_files:
        # Try different potential patterns
        if f"{year}_{month_str}" in file or f"{year}-{month_str}" in file:
            return file

    # If no exact match, try a more flexible match
    for file in job_files:
        if year_str in file and month_str in file:
            return file

    return None


def concat_time_series_files(local_files: List[str], output_file: str) -> bool:
    """Concatenate multiple time series files into a single file"""
    try:
        logger.info(f"Concatenating {len(local_files)} files into {output_file}")

        # Read and concatenate all files using polars
        dfs = []
        for file in local_files:
            try:
                df = pl.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading file {file}: {str(e)}")
                continue

        if not dfs:
            logger.error("No valid dataframes to concatenate")
            return False

        combined_df = pl.concat(dfs)
        combined_df.write_csv(output_file)
        logger.info(f"Successfully concatenated files into {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error concatenating files: {str(e)}")
        return False


def process_year_month(year: int, month: int,
                       timeseries_bucket: str,
                       job_bucket: str,
                       output_bucket: str,
                       temp_dir: str) -> bool:
    """Process data for a specific year and month"""
    try:
        logger.info(f"Processing data for {year}-{month:02d}")

        # List all files in the timeseries bucket
        timeseries_files = list_s3_files(timeseries_bucket)

        # Filter to get only files for this year/month
        relevant_ts_files = []
        for file in timeseries_files:
            file_info = parse_time_series_filename(file)
            if file_info and file_info[0] == year and file_info[1] == month:
                relevant_ts_files.append(file)

        if not relevant_ts_files:
            logger.error(f"No timeseries files found for {year}-{month:02d}")
            return False

        logger.info(f"Found {len(relevant_ts_files)} timeseries files for {year}-{month:02d}")

        # List job files
        job_files = list_s3_files(job_bucket)

        # Find matching job file
        job_file = get_job_filename_for_period(job_files, year, month)
        if not job_file:
            logger.error(f"No job file found for {year}-{month:02d}")
            return False

        logger.info(f"Found job file {job_file} for {year}-{month:02d}")

        # Create local file paths
        local_ts_files = []
        for file in relevant_ts_files:
            local_path = os.path.join(temp_dir, os.path.basename(file))
            if download_s3_file(timeseries_bucket, file, local_path):
                local_ts_files.append(local_path)

        if not local_ts_files:
            logger.error(f"Failed to download any timeseries files for {year}-{month:02d}")
            return False

        local_job_file = os.path.join(temp_dir, os.path.basename(job_file))
        if not download_s3_file(job_bucket, job_file, local_job_file):
            logger.error(f"Failed to download job file {job_file}")
            return False

        # If there are multiple timeseries files, concatenate them
        if len(local_ts_files) > 1:
            combined_ts_file = os.path.join(temp_dir, f"combined_ts_{year}_{month:02d}.csv")
            if not concat_time_series_files(local_ts_files, combined_ts_file):
                logger.error("Failed to concatenate timeseries files")
                return False
        else:
            combined_ts_file = local_ts_files[0]

        # Output file path
        output_file = os.path.join(temp_dir, f"transformed_{year}_{month:02d}.csv")

        # Process the data using the provided function
        logger.info(f"Joining job and timeseries data for {year}-{month:02d}")
        join_job_timeseries(local_job_file, combined_ts_file, output_file)

        # Upload the result
        if not upload_to_s3([output_file], output_bucket):
            logger.error(f"Failed to upload result for {year}-{month:02d}")
            return False

        logger.info(f"Successfully processed data for {year}-{month:02d}")
        return True

    except Exception as e:
        logger.error(f"Error processing {year}-{month:02d}: {str(e)}")
        return False


def main():
    """Main function to process all time series and job data"""
    # Configure bucket names
    timeseries_bucket = "data-transform-conte"
    job_bucket = "conte-job-accounting"
    output_bucket = "conte-transformed"

    logger.info("Starting data transformation process")

    # Create temporary directory for file processing
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory at {temp_dir}")

        # Get list of all timeseries files
        timeseries_files = list_s3_files(timeseries_bucket)
        logger.info(f"Found {len(timeseries_files)} files in timeseries bucket")

        # Extract unique year-month combinations
        periods = set()
        for file in timeseries_files:
            file_info = parse_time_series_filename(file)
            if file_info:
                periods.add(file_info)

        logger.info(f"Found {len(periods)} unique year-month periods to process")

        # Process each year-month combination
        successful = 0
        for year, month in sorted(periods):
            if process_year_month(year, month,
                                  timeseries_bucket,
                                  job_bucket,
                                  output_bucket,
                                  temp_dir):
                successful += 1

        logger.info(f"Completed processing {successful}/{len(periods)} periods")


if __name__ == "__main__":
    main()