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
# - Reads Bronze CSV:  bronze_loan_daily_YYYY_MM_DD.csv
# - Cleans & standardizes schema
# - Adds derived fields: mob, installments_missed, first_missed_date, dpd
# - Writes Parquet:      silver_loan_daily_YYYY_MM_DD.parquet
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
    # mob = months on book (here aligned to installment_num)
    ("mob", IntegerType()),
    # installments_missed (safe divide & ceil)
    ("installments_missed", IntegerType()),
    # first_missed_date
    ("first_missed_date", DateType()),
    # dpd = days past due
    ("dpd", IntegerType()),
    # audit columns
    ("etl_run_ts", None),
    ("etl_source", None),
]

def _build_paths(snapshot_date_str: str, bronze_dir: str, silver_dir: str):
    """Constructs Bronze input path and Silver output path using your existing naming convention."""
    date_token = snapshot_date_str.replace("-", "_")
    bronze_name = f"bronze_loan_daily_{date_token}.csv"
    silver_name = f"silver_loan_daily_{date_token}.parquet"
    bronze_path = os.path.join(bronze_dir, bronze_name)
    silver_path = os.path.join(silver_dir, silver_name)
    return bronze_path, silver_path

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
    If there are accidental duplicates, keep the row with the largest paid_amt,
    then largest overdue_amt as tie-breakers (heuristic).
    """
    w = Window.partitionBy("loan_id", "snapshot_date", "installment_num") \
              .orderBy(col("paid_amt").desc_nulls_last(),
                       col("overdue_amt").desc_nulls_last())

    df = df.withColumn("_rn", F.row_number().over(w)).filter(col("_rn") == 1).drop("_rn")
    return df

def _derive_fields(df):
    # mob aligned to installment_num
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Safe installments_missed: avoid divide by zero; null -> 0
    df = df.withColumn(
        "installments_missed",
        when(col("due_amt").isNull() | (col("due_amt") <= 0), None)
        .otherwise(ceil(col("overdue_amt") / col("due_amt")))
        .cast(IntegerType())
    )

    # Fill null with 0 for downstream logic
    df = df.fillna({"installments_missed": 0, "overdue_amt": 0.0})

    # first_missed_date only when there are missed installments
    df = df.withColumn(
        "first_missed_date",
        when(col("installments_missed") > 0,
             add_months(col("snapshot_date"), -1 * col("installments_missed"))
        ).otherwise(None).cast(DateType())
    )

    # dpd: days past due only when overdue_amt > 0, else 0
    df = df.withColumn(
        "dpd",
        when(col("overdue_amt") > 0.0,
             datediff(col("snapshot_date"), col("first_missed_date"))
        ).otherwise(0).cast(IntegerType())
    )

    # Auditing
    df = df.withColumn("etl_run_ts", current_timestamp())
    df = df.withColumn("etl_source", lit("bronze_loan_daily_csv"))
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
    bronze_lms_directory : path ending with '/' where Bronze CSV is stored
    silver_loan_daily_directory : path ending with '/' where Silver Parquet will be written
    spark : active SparkSession
    fail_on_missing : if True, raises when Bronze file is not present
    """
    # Parse date and build paths
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"[Silver] Invalid snapshot_date '{snapshot_date_str}': {e}")

    bronze_path, silver_path = _build_paths(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory)

    if not os.path.exists(bronze_path):
        msg = f"[Silver] Bronze input not found at {bronze_path}"
        if fail_on_missing:
            raise FileNotFoundError(msg)
        else:
            print(msg)
            return None

    # Load Bronze
    df = spark.read.csv(bronze_path, header=True, inferSchema=True)
    print(f"[Silver] Loaded Bronze: {bronze_path} | rows={df.count()}")

    # Basic validation & filtering on snapshot_date to guard accidental mismatches
    _validate_required_columns(df, REQUIRED_COLUMNS)
    df = _enforce_schema(df)

    # Keep only the requested snapshot_date (in case the CSV contains multiple)
    df = df.filter(col("snapshot_date") == F.lit(snapshot_date_str).cast(DateType()))

    # Deduplicate
    df = _dedupe(df)

    # Derive fields
    df = _derive_fields(df)

    # Reorder columns (required + derived + any extras) for cleanliness
    base_cols = list(SILVER_SCHEMA.keys())
    derived_cols = [c for c, _ in DERIVED_COLUMNS]
    other_cols = [c for c in df.columns if c not in base_cols + derived_cols]
    final_cols = base_cols + derived_cols + other_cols
    df = df.select(*final_cols)

    # Write Silver
    # Keeping your existing single-file-per-snapshot convention for compatibility
    # (You can switch to partitionBy("snapshot_date") if preferred.)
    df.write.mode("overwrite").parquet(silver_path)
    print(f"[Silver] Saved Silver: {silver_path}")

    return df


# ------------------------- CLI ENTRYPOINT -------------------------
def _build_spark(app_name: str = "dev"):
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
    parser.add_argument("--bronze_dir", required=True, help="Directory where Bronze CSV lives, e.g. datamart/bronze/")
    parser.add_argument("--silver_dir", required=True, help="Directory where Silver Parquet will be written, e.g. datamart/silver/")
    args = parser.parse_args()

    spark = _build_spark("silver")
    try:
        process_silver_table(
            snapshot_date_str=args.snapshotdate,
            bronze_lms_directory=args.bronze_dir if args.bronze_dir.endswith(os.sep) else args.bronze_dir + os.sep,
            silver_loan_daily_directory=args.silver_dir if args.silver_dir.endswith(os.sep) else args.silver_dir + os.sep,
            spark=spark,
            fail_on_missing=True
        )
    finally:
        spark.stop()
