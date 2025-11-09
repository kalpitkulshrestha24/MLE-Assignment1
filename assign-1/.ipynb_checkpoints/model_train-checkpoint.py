import os
import json
import argparse
from datetime import datetime
from glob import glob

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

# ---------------------------
# Helpers
# ---------------------------

NUMERIC_TYPES = {"int", "bigint", "double", "float", "decimal", "smallint", "tinyint"}

ID_COLS = {"loan_id", "Customer_ID"}
DATE_COLS = {"snapshot_date", "loan_start_date"}
LABEL_COL = "label"
WEIGHT_COL = "class_weight"


def _resolve_partition_path(base_dir: str, snapshot_date: str, prefix: str, ext: str):
    """
    Resolve a Spark Parquet *folder* written by your Gold step.
    Tries an exact folder:
      {base_dir}/{prefix}_{YYYY_MM_DD}.{ext}
    Else falls back to the newest matching folder in base_dir.
    """
    target = os.path.join(
        base_dir,
        f"{prefix}_{snapshot_date.replace('-','_')}.{ext}"
    )
    if target and os.path.exists(target):
        return target

    # fallback: pick latest matching folder
    pattern = os.path.join(base_dir, f"{prefix}_*.{ext}")
    candidates = sorted(glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files found with pattern: {pattern}")
    return candidates[-1]


def _pick_numeric_features(df):
    feats = []
    for name, dtype in df.dtypes:
        base = dtype.lower().split("(")[0]
        if name in ID_COLS or name in DATE_COLS or name == LABEL_COL:
            continue
        # very conservative: avoid obvious leakage/targets by name
        if any(tok in name.lower() for tok in ["label", "dpd"]):
            continue
        if base in NUMERIC_TYPES:
            feats.append(name)
    return sorted(list(set(feats)))


def _customer_group_split(df, train_pct=0.8, seed=42):
    """
    Group-aware split: same Customer_ID never appears in both splits.
    """
    df = df.withColumn("split_key", F.coalesce(F.col("Customer_ID"), F.col("loan_id")))
    df = df.withColumn("bucket",
                       (F.pmod(F.hash(F.col("split_key").cast(StringType())), F.lit(10_000)).cast("int")))
    threshold = int(train_pct * 10_000)
    train = df.filter(F.col("bucket") < threshold).drop("bucket", "split_key")
    test  = df.filter(F.col("bucket") >= threshold).drop("bucket", "split_key")
    return train, test


def _add_class_weights(df, label_col=LABEL_COL):
    """
    Compute inverse-frequency class weights to mitigate imbalance.
    weight(c) = N / (num_classes * count_c)
    """
    agg = df.groupBy(label_col).count().withColumnRenamed("count", "n")
    stats = {int(r[label_col]): int(r["n"]) for r in agg.collect()}
    total = sum(stats.values())
    num_classes = max(len(stats), 1)
    weights = {int(k): float(total) / (num_classes * v) for k, v in stats.items()}

    # Build create_map(k1, v1, k2, v2, ...)
    mapping_kv = []
    for k, v in weights.items():
        mapping_kv.extend([F.lit(int(k)), F.lit(float(v))])

    mapping = F.create_map(*mapping_kv)
    return df.withColumn(WEIGHT_COL, mapping[F.col(label_col)])


def _confusion_matrix(pred_df, label_col=LABEL_COL, pred_col="prediction"):
    cm = (pred_df
          .groupBy(label_col, pred_col)
          .count()
          .toPandas()
          .pivot(index=label_col, columns=pred_col, values="count")
          .fillna(0)
          .astype(int))
    # Return as dict for JSON
    cm_dict = {int(idx): {int(col): int(cm.loc[idx, col]) for col in cm.columns} for idx in cm.index}
    return cm_dict


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train simple binary classifier on Gold stores.")
    parser.add_argument("--snapshot_date", required=False, default=None, help="YYYY-MM-DD; if not found, use latest partitions")

    # ALIGNED with your Gold writer: gold_base_dir/loan_daily/{feature_store,label_store}/gold_*_{YYYY_MM_DD}.parquet/
    parser.add_argument("--gold_feature_dir", required=False, default="datamart/gold/label_store/feature_store/")
    parser.add_argument("--gold_label_dir",   required=False, default="datamart/gold/label_store/label_store/")

    parser.add_argument("--models_dir",       required=False, default="datamart/models/")
    parser.add_argument("--train_pct",        type=float, default=0.8)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--max_iter",         type=int, default=100)
    args = parser.parse_args()

    spark = (SparkSession.builder
             .appName("CS611-Model-Training")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---------------------------
    # Load Gold tables
    # ---------------------------
    snap = args.snapshot_date or "0000-00-00"  # triggers "latest" fallback when exact not found
    feat_path = _resolve_partition_path(args.gold_feature_dir, snap,
                                        prefix="gold_feature_store", ext="parquet")
    label_path = _resolve_partition_path(args.gold_label_dir, snap,
                                         prefix="gold_label_store", ext="parquet")

    print(f"[INFO] Using feature file: {feat_path}")
    print(f"[INFO] Using label   file: {label_path}")

    features_df = spark.read.parquet(feat_path)
    labels_df   = spark.read.parquet(label_path)

    # Align on keys present in your label gold script
    join_keys = [c for c in ["loan_id", "Customer_ID", "snapshot_date"]
                 if c in features_df.columns and c in labels_df.columns]
    if not join_keys:
        join_keys = ["loan_id"] if "loan_id" in features_df.columns and "loan_id" in labels_df.columns else []
    if not join_keys:
        raise ValueError("Could not find join keys between features and labels.")

    df = (features_df.alias("x")
          .join(labels_df.select(*(join_keys + [LABEL_COL, "label_def"])).alias("y"), on=join_keys, how="inner"))

    joined_rows = df.count()
    print(f"[INFO] Joined dataset rows: {joined_rows}")
    if joined_rows == 0:
        raise ValueError("Joined dataset is empty. Check that feature and label snapshots match the same date.")

    # ---------------------------
    # Feature selection & prep
    # ---------------------------
    numeric_feats = _pick_numeric_features(df)
    if not numeric_feats:
        raise ValueError("No numeric features detected. Ensure your Gold feature store has numeric columns.")

    print(f"[INFO] Using {len(numeric_feats)} numeric features.")

    # Impute missing numerics
    imputer = Imputer(inputCols=numeric_feats, outputCols=[f"{c}__imp" for c in numeric_feats])

    # Assemble into vector
    assembler = VectorAssembler(
        inputCols=[f"{c}__imp" for c in numeric_feats],
        outputCol="features"
    )

    # Class weights
    df = _add_class_weights(df, label_col=LABEL_COL)

    # ---------------------------
    # Train / Test split (group-aware)
    # ---------------------------
    train_df, test_df = _customer_group_split(df, train_pct=args.train_pct, seed=args.seed)
    print(f"[INFO] Train rows: {train_df.count()} | Test rows: {test_df.count()}")

    # ---------------------------
    # Model
    # ---------------------------
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=args.max_iter,
        regParam=0.0,
        elasticNetParam=0.0
    )

    pipeline = Pipeline(stages=[imputer, assembler, lr])
    model = pipeline.fit(train_df)

    # ---------------------------
    # Evaluate
    # ---------------------------
    pred = model.transform(test_df)

    evaluator_roc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    auc_roc = evaluator_roc.evaluate(pred)
    auc_pr  = evaluator_pr.evaluate(pred)

    # probability[1] to scalar, then default 0.5 threshold
    pred = pred.withColumn("p1", vector_to_array("probability").getItem(1))
    pred = pred.withColumn("prediction", (F.col("p1") >= F.lit(0.5)).cast("int"))
    cm = _confusion_matrix(pred.select(LABEL_COL, "prediction"))

    print(f"[METRIC] AUC-ROC: {auc_roc:.4f}")
    print(f"[METRIC] AUC-PR : {auc_pr:.4f}")
    print(f"[METRIC] Confusion Matrix: {cm}")

    # ---------------------------
    # Persist model + metrics
    # ---------------------------
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.models_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "spark_lr_model")
    model.write().overwrite().save(model_path)

    # Safer label_def extraction without pandas dependency on entire df
    label_def_val = None
    if "label_def" in df.columns:
        first_row = df.select("label_def").dropna().limit(1).collect()
        if first_row:
            label_def_val = first_row[0]["label_def"]

    metrics = {
        "run_id": run_id,
        "snapshot_feature_path": feat_path,
        "snapshot_label_path": label_path,
        "train_rows": train_df.count(),
        "test_rows": test_df.count(),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "confusion_matrix": cm,
        "label_def": label_def_val,
        "feature_count": len(numeric_feats),
        "features_used": numeric_feats,
        "notes": "Customer-grouped split; inverse-frequency class weights; 0.5 threshold."
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved metrics to: {os.path.join(out_dir, 'metrics.json')}")
    spark.stop()


if __name__ == "__main__":
    main()
