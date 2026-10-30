# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — synthetic data
# MAGIC
# MAGIC Generates the two sources of the PIX fraud model as JSON files in the landing UC Volume of the target schema:
# MAGIC
# MAGIC | Folder | Content |
# MAGIC |---|---|
# MAGIC | `pix_transactions/` | ~1M PIX transactions from ~10k holders in the 45 days before the run (source of the batch Feature Views) |
# MAGIC | `fraud_labels/` | ~100k transactions sampled from the last 7 days, one per anchor (`holder_id`, `created_at_brt`), with ~5% fraud |
# MAGIC
# MAGIC The `ingest` pipeline (Lakeflow SDP) loads these files into the `raw_pix_transactions` and `raw_fraud_labels` tables.
# MAGIC
# MAGIC The generation is deterministic (fixed seed) and idempotent: a folder that already has files is not regenerated.
# MAGIC `created_at_brt` is written as Brasília local time without offset (`yyyy-MM-dd'T'HH:mm:ss`).

# COMMAND ----------

dbutils.widgets.text("landing", "")  # volume path, e.g. /Volumes/<catalog>/<schema>/landing

LANDING = dbutils.widgets.get("landing")
TRANSACTIONS = f"{LANDING}/pix_transactions"
LABELS = f"{LANDING}/fraud_labels"
TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss"

SEED = 42
N_HOLDERS = 10_000
N_TRANSACTIONS = 1_000_000
N_LABELS = 100_000
DAYS = 45          # transaction period
ANCHOR_DAYS = 7    # anchors in the last days of the period (30d window always complete)
FRAUD_RATE = 0.05

# COMMAND ----------

# MAGIC %md
# MAGIC ## PIX transactions
# MAGIC
# MAGIC - `holder_id`: `H` + 10 digits; activity is concentrated in part of the holders (skewed distribution).
# MAGIC - `created_at_brt`: uniform instant in the 45 days before the run date (Brasília time).
# MAGIC - `amount`: log-normal value (median ~R$ 90).

# COMMAND ----------

from datetime import datetime, timedelta, timezone
from pyspark.sql import functions as F, Window

def has_files(path):
    try:
        return len(dbutils.fs.ls(path)) > 0
    except Exception:  # folder does not exist yet
        return False

if not has_files(TRANSACTIONS):
    today_brt = datetime.now(timezone(timedelta(hours=-3))).date()
    start = F.lit(str(today_brt - timedelta(days=DAYS))).cast("timestamp")
    holder_idx = (F.pow(F.rand(SEED), 1.5) * N_HOLDERS).cast("long")
    transactions = (
        spark.range(0, N_TRANSACTIONS, numPartitions=8)
        .select(
            F.format_string("H%010d", F.pmod(F.xxhash64(holder_idx), F.lit(10**10))).alias("holder_id"),
            F.timestamp_seconds(F.unix_timestamp(start) + (F.rand(SEED + 1) * DAYS * 86400).cast("long")).alias("created_at_brt"),
            F.round(F.exp(F.randn(SEED + 2) + 4.5), 2).alias("amount"),
        )
    )
    transactions.write.option("timestampFormat", TS_FORMAT).json(TRANSACTIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fraud labels
# MAGIC
# MAGIC The anchors are actual transactions of `pix_transactions/` from the last 7 days, without repeated (`holder_id`, `created_at_brt`).
# MAGIC The `label` has a synthetic signal: the score grows with the `amount` and with the number of transfers of the holder in the previous 24h, plus deterministic noise; the top 5% scores are fraud.

# COMMAND ----------

if not has_files(LABELS):
    transactions = (
        spark.read.schema("holder_id STRING, created_at_brt TIMESTAMP, amount DOUBLE")
        .option("timestampFormat", TS_FORMAT).json(TRANSACTIONS)
    )
    end = transactions.select(F.date_add(F.to_date(F.max("created_at_brt")), 1)).first()[0]
    anchor_start = F.lit(str(end - timedelta(days=ANCHOR_DAYS))).cast("timestamp")

    last_24h = Window.partitionBy("holder_id").orderBy(F.col("created_at_brt").cast("long")).rangeBetween(-86400, 0)
    noise_u = (F.pmod(F.xxhash64("holder_id", "created_at_brt", F.lit(SEED)), F.lit(1_000_000)) + 0.5) / 1_000_000

    anchors = (
        transactions
        .withColumn("transfers_24h", F.count("*").over(last_24h))
        .where(F.col("created_at_brt") >= anchor_start)
        .dropDuplicates(["holder_id", "created_at_brt"])
        .orderBy(F.xxhash64("holder_id", "created_at_brt"))
        .limit(N_LABELS)
        .withColumn("score", F.log("amount") + F.log("transfers_24h") + 1.5 * F.log(noise_u / (1 - noise_u)))
    )
    threshold = anchors.select(F.percentile_approx("score", 1 - FRAUD_RATE, 10_000)).first()[0]
    labels = anchors.select(
        "holder_id", "created_at_brt", "amount",
        (F.col("score") >= threshold).cast("int").alias("label"),
    )
    labels.write.option("timestampFormat", TS_FORMAT).json(LABELS)
