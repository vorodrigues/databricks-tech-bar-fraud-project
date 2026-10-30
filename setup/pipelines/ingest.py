# Ingestion pipeline (Lakeflow Spark Declarative Pipelines) of the PIX fraud model sources.
#
# Flow: landing UC Volume (JSON) → bronze (streaming tables, Auto Loader) → silver (materialized views, expectations).
# - Bronze keeps the file fields as read, plus the source file and the ingestion instant.
# - Silver casts to the contract consumed by features, training and scoring, and drops invalid rows.
#
# Reference: https://docs.databricks.com/aws/en/ldp/

from pyspark import pipelines as dp
from pyspark.sql import functions as F

LANDING = spark.conf.get("landing")  # volume path, set in the pipeline configuration
TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss"  # Brasília local time without offset


def auto_loader(folder):
    """Incremental read of the new JSON files of a landing folder."""
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .load(f"{LANDING}/{folder}")
        .select("*", F.col("_metadata.file_path").alias("source_file"), F.current_timestamp().alias("ingested_at"))
    )


# Bronze

@dp.table(comment="PIX transactions as landed (JSON), with source file and ingestion instant")
def bronze_pix_transactions():
    return auto_loader("pix_transactions")


@dp.table(comment="Fraud labels as landed (JSON), with source file and ingestion instant")
def bronze_fraud_labels():
    return auto_loader("fraud_labels")


# Silver

@dp.materialized_view(comment="PIX transactions: holder_id, created_at_brt (Brasília time), amount")
@dp.expect_all_or_drop({
    "valid_holder_id": "holder_id RLIKE '^H[0-9]{10}$'",
    "valid_created_at_brt": "created_at_brt IS NOT NULL",
    "positive_amount": "amount > 0",
})
def raw_pix_transactions():
    return spark.read.table("bronze_pix_transactions").select(
        F.col("holder_id").cast("string").alias("holder_id"),
        F.to_timestamp("created_at_brt", TS_FORMAT).alias("created_at_brt"),
        F.col("amount").cast("double").alias("amount"),
    )


@dp.materialized_view(comment="Fraud labels: one anchor (holder_id, created_at_brt) per row, label 1 = fraud")
@dp.expect_all_or_drop({
    "valid_keys": "holder_id IS NOT NULL AND created_at_brt IS NOT NULL",
    "valid_label": "label IN (0, 1)",
})
def raw_fraud_labels():
    return spark.read.table("bronze_fraud_labels").select(
        F.col("holder_id").cast("string").alias("holder_id"),
        F.to_timestamp("created_at_brt", TS_FORMAT).alias("created_at_brt"),
        F.col("amount").cast("double").alias("amount"),
        F.col("label").cast("int").alias("label"),
    )
