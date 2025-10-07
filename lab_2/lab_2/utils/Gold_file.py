# utils/data_processing_gold_table.py

import os
from datetime import datetime
from typing import Optional

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    StringType, IntegerType, FloatType, DateType
)

# -----------------------------
# LABEL STORE (kept from your current code, minor cleanups)
# -----------------------------
def process_labels_gold_table(
    snapshot_date_str: str,
    silver_loan_daily_directory: str,
    gold_label_store_directory: str,
    spark: SparkSession,
    dpd: int,
    mob: int
) -> DataFrame:
    """
    Build gold label store (binary label) from silver loan-daily snapshot.
    Label = 1 if dpd >= dpd_threshold at given mob, else 0.
    Outputs: gold_label_store_<YYYY_MM_DD>.parquet
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Load silver table partition for the snapshot
    silver_part = f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    silver_path = os.path.join(silver_loan_daily_directory, silver_part)
    df = spark.read.parquet(silver_path)

    # Restrict to the target MOB at the snapshot
    df = df.filter(F.col("mob") == F.lit(mob))

    # Create label
    df = (
        df.withColumn("label", F.when(F.col("dpd") >= F.lit(dpd), F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()))
          .withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    )

    # Select final columns
    out = df.select(
        "loan_id",
        "Customer_ID",
        "label",
        "label_def",
        "snapshot_date"
    )

    # Write gold label store
    os.makedirs(gold_label_store_directory, exist_ok=True)
    out_part = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(gold_label_store_directory, out_part)
    out.write.mode("overwrite").parquet(out_path)

    return out


# -----------------------------
# FEATURE STORE (new)
# -----------------------------
def process_features_gold_table(
    snapshot_date_str: str,
    silver_loan_daily_directory: str,
    gold_feature_store_directory: str,
    spark: SparkSession,
    mob: Optional[int] = None
) -> DataFrame:
    """
    Build a gold feature store from the silver loan-daily snapshot.
    Produces engineered, ML-ready features at (loan_id, snapshot_date) granularity.
    If `mob` is provided, restricts features to that MOB (keeps parity with label store usage).

    Outputs: gold_feature_store_<YYYY_MM_DD>.parquet
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Load silver table partition for the snapshot
    silver_part = f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    silver_path = os.path.join(silver_loan_daily_directory, silver_part)
    s = spark.read.parquet(silver_path)

    # Optional: align the feature cohort to the same MOB as labels (common for training sets)
    if mob is not None:
        s = s.filter(F.col("mob") == F.lit(mob))

    # Defensive casts (idempotent if already correct)
    type_map = {
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
        "mob": IntegerType(),
        "installments_missed": IntegerType(),
        "dpd": IntegerType(),
    }
    for c, t in type_map.items():
        if c in s.columns:
            s = s.withColumn(c, F.col(c).cast(t))

    # ----- Core numeric features -----
    features = (
        s
        .withColumn("tenure_remaining", (F.col("tenure") - F.col("installment_num")).cast(IntegerType()))
        .withColumn("paid_to_due_ratio",
                    F.when(F.col("due_amt") > 0, F.col("paid_amt") / F.col("due_amt")).otherwise(F.lit(0.0)))
        .withColumn("overdue_to_due_ratio",
                    F.when(F.col("due_amt") > 0, F.col("overdue_amt") / F.col("due_amt")).otherwise(F.lit(0.0)))
        .withColumn("balance_to_loan_ratio",
                    F.when(F.col("loan_amt") > 0, F.col("balance") / F.col("loan_amt")).otherwise(F.lit(0.0)))
        .withColumn("is_early_tenure", F.when(F.col("mob") <= 3, F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()))
        .withColumn("is_mid_tenure",   F.when((F.col("mob") > 3) & (F.col("mob") <= 12), F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()))
        .withColumn("is_late_tenure",  F.when(F.col("mob") > 12, F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()))
        .withColumn("risk_flag_high_dpd", F.when(F.col("dpd") >= 30, F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()))
        # simple bucketing can help tree models and avoids leakage if defined off current snapshot only
        .withColumn("utilization_bucket", F.when(F.col("balance_to_loan_ratio") >= 0.9, F.lit(">=90%"))
                                           .when(F.col("balance_to_loan_ratio") >= 0.7, F.lit("70–90%"))
                                           .when(F.col("balance_to_loan_ratio") >= 0.5, F.lit("50–70%"))
                                           .when(F.col("balance_to_loan_ratio") >  0.0, F.lit("0–50%"))
                                           .otherwise(F.lit("0% or NA")))
        .withColumn("overdue_bucket", F.when(F.col("dpd") >= 60, F.lit("60+"))
                                       .when(F.col("dpd") >= 30, F.lit("30–59"))
                                       .when(F.col("dpd") >= 1,  F.lit("1–29"))
                                       .otherwise(F.lit("0")))
    )

    # Select a clean ML-ready set (loan-level)
    out = features.select(
        # keys
        "loan_id", "Customer_ID", "snapshot_date",
        # base signals
        "mob", "tenure", "tenure_remaining", "installment_num",
        "loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance", "dpd", "installments_missed",
        # engineered ratios
        "paid_to_due_ratio", "overdue_to_due_ratio", "balance_to_loan_ratio",
        # tenure flags
        "is_early_tenure", "is_mid_tenure", "is_late_tenure",
        # risk proxies / buckets
        "risk_flag_high_dpd", "utilization_bucket", "overdue_bucket",
    )

    # Write gold feature store
    os.makedirs(gold_feature_store_directory, exist_ok=True)
    out_part = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(gold_feature_store_directory, out_part)
    out.write.mode("overwrite").parquet(out_path)

    return out


# -----------------------------
# TRAINING SET (features ⨝ labels)
# -----------------------------
def build_gold_training_set(
    snapshot_date_str: str,
    gold_feature_store_directory: str,
    gold_label_store_directory: str,
    gold_training_set_directory: str,
    spark: SparkSession
) -> DataFrame:
    """
    Join the gold feature store with the gold label store on (loan_id, snapshot_date)
    to produce a single, model-ready training set.
    Outputs: gold_training_set_<YYYY_MM_DD>.parquet
    """
    feat_part = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    label_part = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"

    feat_path = os.path.join(gold_feature_store_directory, feat_part)
    label_path = os.path.join(gold_label_store_directory, label_part)

    feats = spark.read.parquet(feat_path)
    labels = spark.read.parquet(label_path)

    train = (
        feats.alias("x")
        .join(labels.select("loan_id", "snapshot_date", "label", "label_def").alias("y"),
              on=["loan_id", "snapshot_date"],
              how="inner")
    )

    os.makedirs(gold_training_set_directory, exist_ok=True)
    out_part = f"gold_training_set_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(gold_training_set_directory, out_part)
    train.write.mode("overwrite").parquet(out_path)
    return train


# -----------------------------
# (Optional) simple local test entrypoint
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gold layer: feature store, label store, and training set.")
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    parser.add_argument("--silver_dir", default="datamart/silver/")
    parser.add_argument("--gold_label_dir", default="datamart/gold/label_store/")
    parser.add_argument("--gold_feature_dir", default="datamart/gold/feature_store/")
    parser.add_argument("--gold_training_dir", default="datamart/gold/training_set/")
    parser.add_argument("--dpd", type=int, default=30, help="DPD threshold for label")
    parser.add_argument("--mob", type=int, default=6, help="Target MOB for cohort")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("gold").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Build label + features + training set
    process_labels_gold_table(
        snapshot_date_str=args.snapshotdate,
        silver_loan_daily_directory=args.silver_dir,
        gold_label_store_directory=args.gold_label_dir,
        spark=spark,
        dpd=args.dpd,
        mob=args.mob
    )

    process_features_gold_table(
        snapshot_date_str=args.snapshotdate,
        silver_loan_daily_directory=args.silver_dir,
        gold_feature_store_directory=args.gold_feature_dir,
        spark=spark,
        mob=args.mob
    )

    build_gold_training_set(
        snapshot_date_str=args.snapshotdate,
        gold_feature_store_directory=args.gold_feature_dir,
        gold_label_store_directory=args.gold_label_dir,
        gold_training_set_directory=args.gold_training_dir,
        spark=spark
    )

    print("Gold layer complete.")
