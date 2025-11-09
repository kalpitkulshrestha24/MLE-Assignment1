# utils/data_processing_silver_table.py

import os
from datetime import datetime
import argparse

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, lit, when, ceil, datediff, add_months, current_timestamp
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# ---------------------------------------------------------------------
# Silver Layer Processor
# - Reads Bronze *Parquet partition*:  <bronze_dir>/snapshot_date_part=YYYY-MM-DD/
# - Cleans & standardizes schema
# - Adds derived fields: mob, installments_missed, first_missed_date, dpd
# - Writes Parquet folder per snapshot (same naming style you used)
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "loan_id", "Customer_ID", "loan_start_date", "tenure",
    "installment_num", "loan_amt", "due_amt", "paid_amt",
    "overdue_amt", "balance", "snapshot_date"
]

SILVER_SCHEMA = {
    "loan_id": StringType(),
    "Customer_ID": StringType(),
    "loan_start_date": DateType(),
    "tenure": IntegerType(),
    "installment_num": IntegerType(),
    "loan_amt": FloatType(),
    "due_amt": FloatType(),
    "paid_amt": FloatType(),
    "overdue_amt": FloatType(),
    "balance": FloatType(),
    "snapshot_date": DateType(),
}

DERIVED_COLUMNS = [
    ("mob", IntegerType()),
    ("installments_missed", IntegerType()),
    ("first_missed_date", DateType()),
    ("dpd", IntegerType()),
    ("etl_run_ts", None),
    ("etl_source", None),
]

def _build_paths(snapshot_date_str: str, bronze_dir: str, silver_dir: str):
    """
    Build Bronze *partition* path and Silver output path (unchanged style).
    Bronze (actual): <bronze_dir>/snapshot_date_part=YYYY-MM-DD/
    Silver (as you had): <silver_dir>/silver_loan_daily_YYYY_MM_DD.parquet
    """
    # Bronze partition folder
    bronze_part = os.path.join(bronze_dir, f"snapshot_date_part={snapshot_date_str}")

    # Keep your existing Silver naming style
    date_token = snapshot_date_str.replace("-", "_")
    silver_name = f"silver_loan_daily_{date_token}.parquet"
    silver_path = os.path.join(silver_dir, silver_name)

    return bronze_part, silver_path

def _validate_required_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"[Silver] Missing required columns from Bronze: {missing}")

def _enforce_schema(df):
    for c, t in SILVER_SCHEMA.items():
        df = df.withColumn(c, col(c).cast(t))
    return df

def _dedupe(df):
    """
    Deduplicate on (loan_id, snapshot_date, installment_num).
    Keep the row with the largest paid_amt, then largest overdue_amt as a tie-breaker.
    """
    w = (
        Window.partitionBy("loan_id", "snapshot_date", "installment_num")
        .orderBy(col("paid_amt").desc_nulls_last(), col("overdue_amt").desc_nulls_last())
    )
    return df.withColumn("_rn", F.row_number().over(w)).filter(col("_rn") == 1).drop("_rn")

def _derive_fields(df):
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    df = df.withColumn(
        "installments_missed",
        when(col("due_amt").isNull() | (col("due_amt") <= 0), None)
        .otherwise(ceil(col("overdue_amt") / col("due_amt")))
        .cast(IntegerType())
    ).fillna({"installments_missed": 0, "overdue_amt": 0.0})

    df = df.withColumn(
        "first_missed_date",
        when(col("installments_missed") > 0, add_months(col("snapshot_date"), -1 * col("installments_missed")))
        .otherwise(None).cast(DateType())
    )

    df = df.withColumn(
        "dpd",
        when(col("overdue_amt") > 0.0, datediff(col("snapshot_date"), col("first_missed_date")))
        .otherwise(0).cast(IntegerType())
    )

    df = df.withColumn("etl_run_ts", current_timestamp())
    df = df.withColumn("etl_source", lit("bronze_parquet_partition"))
    return df

def process_silver_table(snapshot_date_str: str,
                         bronze_lms_directory: str,
                         silver_loan_daily_directory: str,
                         spark: pyspark.sql.SparkSession,
                         fail_on_missing: bool = True):
    """
    Transform Bronze -> Silver.

    Parameters
    ----------
    snapshot_date_str : 'YYYY-MM-DD'
    bronze_lms_directory : base path where Bronze partitions live, e.g. datamart/bronze/lms/
    silver_loan_daily_directory : base path where Silver Parquet will be written
    spark : active SparkSession
    fail_on_missing : if True, raises when Bronze partition is not present
    """
    # Validate date
    try:
        _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"[Silver] Invalid snapshot_date '{snapshot_date_str}': {e}")

    bronze_part_path, silver_path = _build_paths(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory)

    # Existence check (partition folder)
    if not os.path.isdir(bronze_part_path):
        msg = f"[Silver] Bronze input partition not found at {bronze_part_path}"
        if fail_on_missing:
            raise FileNotFoundError(msg)
        else:
            print(msg)
            return None

    # Load Bronze as Parquet (matches your Bronze output)
    df = spark.read.parquet(bronze_part_path)
    print(f"[Silver] Loaded Bronze partition: {bronze_part_path} | rows={df.count()}")

    # Basic validation & types
    _validate_required_columns(df, REQUIRED_COLUMNS)
    df = _enforce_schema(df)

    # Keep only the requested snapshot_date (safety guard)
    df = df.filter(col("snapshot_date") == F.lit(snapshot_date_str).cast(DateType()))

    # Deduplicate
    df = _dedupe(df)

    # Derive fields
    df = _derive_fields(df)

    # Reorder columns (required + derived + any extras)
    base_cols = list(SILVER_SCHEMA.keys())
    derived_cols = [c for c, _ in DERIVED_COLUMNS]
    other_cols = [c for c in df.columns if c not in base_cols + derived_cols]
    final_cols = base_cols + derived_cols + other_cols
    df = df.select(*final_cols)

    # Write Silver (same “one folder per snapshot” naming you used)
    df.write.mode("overwrite").parquet(silver_path)
    print(f"[Silver] Saved Silver: {silver_path}")

    return df

# ------------------------- CLI ENTRYPOINT -------------------------
def _build_spark(app_name: str = "silver"):
    spark = (
        pyspark.sql.SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Silver Loan Daily table")
    parser.add_argument("--snapshotdate", required=True, help="Snapshot date in YYYY-MM-DD")
    parser.add_argument("--bronze_dir", required=True, help="Base dir where Bronze partitions live, e.g. datamart/bronze/lms/")
    parser.add_argument("--silver_dir", required=True, help="Base dir where Silver Parquet will be written, e.g. datamart/silver/lms/")
    args = parser.parse_args()

    # Ensure trailing slash behavior is consistent
    bronze_dir = args.bronze_dir if args.bronze_dir.endswith(os.sep) else args.bronze_dir + os.sep
    silver_dir = args.silver_dir if args.silver_dir.endswith(os.sep) else args.silver_dir + os.sep

    spark = _build_spark("silver")
    try:
        process_silver_table(
            snapshot_date_str=args.snapshotdate,
            bronze_lms_directory=bronze_dir,
            silver_loan_daily_directory=silver_dir,
            spark=spark,
            fail_on_missing=True
        )
    finally:
        spark.stop()
