import os
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import pyspark
from pyspark.sql import SparkSession

from utils.Bronze_file import process_bronze_table
from utils.Silver_file import process_silver_table
from utils.data_processing_gold_table import process_gold_table



def generate_first_of_month_dates(start_date_str: str, end_date_str: str):
    """Return list of yyyy-mm-dd strings for the 1st of each month in [start, end]."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(day=1)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur = cur + relativedelta(months=1)
    return dates


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Medallion pipeline orchestrator")
    p.add_argument("--start_date", default="2023-01-01", help="inclusive, YYYY-MM-DD")
    p.add_argument("--end_date",   default="2024-12-01", help="inclusive, YYYY-MM-DD")
    p.add_argument("--dpd", type=int, default=30, help="label threshold: days past due")
    p.add_argument("--mob", type=int, default=6, help="label snapshot: months on book")
    p.add_argument("--bronze_dir", default="datamart/bronze/lms/", help="bronze output dir")
    p.add_argument("--silver_dir", default="datamart/silver/loan_daily/", help="silver output dir")
    p.add_argument("--gold_dir",   default="datamart/gold/label_store/", help="gold output dir")
    return p


def main():
    args = build_arg_parser().parse_args()

    # --- Spark session ---
    spark: SparkSession = (
        pyspark.sql.SparkSession.builder
        .appName("dev")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # --- Dates to process ---
    dates_str_lst = generate_first_of_month_dates(args.start_date, args.end_date)
    print(f"[INFO] Processing snapshots: {dates_str_lst}")

    # --- Ensure output dirs ---
    ensure_dir(args.bronze_dir)
    ensure_dir(args.silver_dir)
    ensure_dir(args.gold_dir)

    # --- Bronze backfill ---
    print("\n==== BRONZE ====")
    for snapshot_date_str in dates_str_lst:
        # expected signature in Bronze_file.py:
        # def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
        process_bronze_table(snapshot_date_str, args.bronze_dir, spark)

    # --- Silver backfill ---
    print("\n==== SILVER ====")
    for snapshot_date_str in dates_str_lst:
        # expected signature in Silver_file.py:
        # def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
        process_silver_table(snapshot_date_str, args.bronze_dir, args.silver_dir, spark)

    # --- Gold backfill (labels) ---
    print("\n==== GOLD ====")
    for snapshot_date_str in dates_str_lst:
        # expected signature in Gold_file.py (implement this in your file):
        # def process_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
        process_gold_table(snapshot_date_str, args.silver_dir, args.gold_dir, spark, dpd=args.dpd, mob=args.mob)

    # --- Sanity check: read all gold outputs & show count ---
    print("\n==== VERIFY GOLD OUTPUT ====")
    files = [args.gold_dir + os.path.basename(f) for f in glob.glob(os.path.join(args.gold_dir, "*"))]
    if files:
        df = spark.read.option("header", "true").parquet(*files)
        print("gold_row_count:", df.count())
        df.show(10, truncate=False)
    else:
        print("[WARN] No gold files found at:", args.gold_dir)

    spark.stop()


if __name__ == "__main__":
    main()
