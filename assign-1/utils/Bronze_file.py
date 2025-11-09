# bronze_file.py
# Ingest raw LMS loan daily CSV into Bronze with lineage + partitioned output.

import argparse
from datetime import datetime
import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType
)

def get_schema():
    # Keep Bronze raw: store as strings to avoid type drift; parse later in Silver.
    return StructType([
        StructField("loan_id",        StringType(),  True),
        StructField("Customer_ID",    StringType(),  True),
        StructField("loan_start_date",StringType(),  True),
        StructField("tenure",         StringType(),  True),
        StructField("installment_num",StringType(),  True),
        StructField("loan_amt",       StringType(),  True),
        StructField("due_amt",        StringType(),  True),
        StructField("paid_amt",       StringType(),  True),
        StructField("overdue_amt",    StringType(),  True),
        StructField("balance",        StringType(),  True),
        StructField("snapshot_date",  StringType(),  True),
    ])

def process_bronze_table(
    snapshot_date_str: str,
    bronze_lms_directory: str,
    spark: pyspark.sql.SparkSession,
    source_csv: str = "data/lms_loan_daily.csv",
    fmt: str = "parquet",
):
    """
    Library entrypoint used by main.py
    """
    # validate date
    try:
        snap_dt = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"snapshot_date must be YYYY-MM-DD; got {snapshot_date_str}") from e

    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    print("\n--- Bronze Ingest Start ---")
    print(f"Source CSV    : {source_csv}")
    print(f"Snapshot date : {snapshot_date_str}")
    print(f"Output dir    : {bronze_lms_directory}")
    print(f"Format        : {fmt}")

    schema = get_schema()
    df_raw = (
        spark.read
        .option("header", True)
        .schema(schema)
        .csv(source_csv)
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_ingestion_ts", F.current_timestamp())
    )

    df = (
        df_raw
        .withColumn("snapshot_date_norm", F.to_date(F.col("snapshot_date")))
        .filter(F.col("snapshot_date_norm") == F.lit(snap_dt.isoformat()))
        .withColumn("snapshot_date_part", F.col("snapshot_date_norm").cast(DateType()))
    )

    total_raw = df_raw.count()
    total_filtered = df.count()
    print(f"Rows read (all snapshots): {total_raw}")
    print(f"Rows kept (snapshot={snapshot_date_str}): {total_filtered}")

    out_base = bronze_lms_directory if bronze_lms_directory.endswith("/") else bronze_lms_directory + "/"

    writer = df.write.mode("overwrite").partitionBy("snapshot_date_part")

    if fmt.lower() == "parquet":
        writer.format("parquet").save(out_base)
        print(f"Bronze saved (Parquet partitions) to: {out_base}")
    else:
        writer.option("header", True).format("csv").save(out_base)
        print(f"Bronze saved (CSV partitions) to: {out_base}")

    print("Partition column: snapshot_date_part (YYYY-MM-DD)")
    print("--- Bronze Ingest Done ---\n")

    # return something useful to the orchestrator
    return {"rows_kept": total_filtered, "output_dir": out_base, "format": fmt.lower()}

# ---- Keep the CLI as a thin wrapper so you can also run this module directly ----

def parse_args():
    p = argparse.ArgumentParser(description="Bronze ingest: lms_loan_daily CSV -> Bronze table")
    p.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD snapshot to ingest")
    p.add_argument("--source_csv", default="data/lms_loan_daily.csv", help="Path to raw CSV")
    p.add_argument("--output_dir", default="datamart/bronze/loan_daily/", help="Output base directory")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output file format")
    p.add_argument("--appname", default="bronze_ingest_loan_daily", help="Spark app name")
    return p.parse_args()

def main():
    args = parse_args()
    spark = (
        pyspark.sql.SparkSession.builder
        .appName(args.appname)
        .master("local[*]")
        .getOrCreate()
    )
    try:
        process_bronze_table(
            snapshot_date_str=args.snapshotdate,
            bronze_lms_directory=args.output_dir,
            spark=spark,
            source_csv=args.source_csv,
            fmt=args.format,
        )
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
