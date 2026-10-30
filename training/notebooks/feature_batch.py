# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Feature engineering / Batch
# MAGIC
# MAGIC Task `feature_batch` of the `fraud_training` job. Ensures, **idempotently**, the
# MAGIC **declarative batch Feature Views** and their **managed materializations** (offline Delta + online
# MAGIC Lakebase), ending with the offline table populated for training.
# MAGIC
# MAGIC ## Flow
# MAGIC 1. **Declarative features** (`fe.create_feature`) over `DeltaTableSource`: sum and count of
# MAGIC    `amount` per `holder_id` in 1d, 7d and 30d windows. Reuses the ones already registered (`get_feature`).
# MAGIC 2. **Materialization** (`fe.materialize_features`): managed pipeline that writes the offline store and
# MAGIC    syncs the online store (`TableTrigger` updates on every source change). Created only for
# MAGIC    features without a materialization — rerunning does not duplicate pipelines.
# MAGIC 3. **Waits** for the offline table to have data (asynchronous backfill on the first materialization).
# MAGIC
# MAGIC Dependencies through the serverless environment of the task (`databricks-feature-engineering>=0.16.0`).
# MAGIC Prerequisites (ensured by the `fraud_setup` job): source table `raw_pix_transactions` and
# MAGIC online store shared across targets (`online_store`).

# COMMAND ----------

# DBTITLE 1,Parameters
import time
from datetime import timedelta

# Job parameters arrive as widgets (values of the bundle target)
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("online_store", "")

CATALOG        = dbutils.widgets.get("catalog")
SCHEMA         = dbutils.widgets.get("schema")
RAW_TABLE      = "raw_pix_transactions"        # raw transaction source (event by event)
ONLINE_STORE   = dbutils.widgets.get("online_store")  # existing Lakebase online store
OFFLINE_PREFIX = "batch_pix_features"          # prefix of the managed offline table
ONLINE_PREFIX  = "batch_pix_features_online"   # prefix of the managed online table

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Declarative Feature Views — `batch_pix_{amount,transfers}_{1,7,30}d`
# MAGIC `AggregationFunction(Op(input="amount"), window)` over the Delta source, with `holder_id` as the entity and
# MAGIC `created_at_brt` as the time column (point-in-time). 1d uses `TumblingWindow` (`SlidingWindow` requires
# MAGIC slide < window); 7d/30d use `SlidingWindow` with a daily slide.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities import (
    DeltaTableSource, AggregationFunction, Sum, Count,
    SlidingWindow, TumblingWindow, OfflineStoreConfig, OnlineStoreConfig, TableTrigger)

FEATURE_DEFS = [("batch_pix_amount", Sum), ("batch_pix_transfers", Count)]  # (prefix, operator)
WINDOWS = [1, 7, 30]                                                        # days

fe = FeatureEngineeringClient()
source = DeltaTableSource(catalog_name=CATALOG, schema_name=SCHEMA, table_name=RAW_TABLE)

def window(days):
    if days == 1:
        return TumblingWindow(window_duration=timedelta(days=1))
    return SlidingWindow(window_duration=timedelta(days=days), slide_duration=timedelta(days=1))

def ensure_feature(name, Op, days):
    try:
        return fe.get_feature(full_name=f"{CATALOG}.{SCHEMA}.{name}")  # already registered: reuse
    except Exception:
        return fe.create_feature(
            source=source,
            function=AggregationFunction(Op(input="amount"), window(days)),
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            entity=["holder_id"],
            timeseries_column="created_at_brt",
            name=name)

features = [ensure_feature(f"{prefix}_{w}d", Op, w) for prefix, Op in FEATURE_DEFS for w in WINDOWS]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Managed materialization — offline (Delta) + online (Lakebase)
# MAGIC `fe.list_materialized_features` lists the materializations of each feature; `materialize_features` is only
# MAGIC called for the ones without any (calling it again would create a duplicate pipeline). The table
# MAGIC names are generated (prefix + random suffix) and are therefore always discovered through the API.

# COMMAND ----------

missing = [f for f in features if not fe.list_materialized_features(feature_name=f.full_name)]
if missing:
    fe.materialize_features(
        features=missing,
        offline_config=OfflineStoreConfig(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            table_name_prefix=OFFLINE_PREFIX),
        online_config=OnlineStoreConfig(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            table_name_prefix=ONLINE_PREFIX,
            online_store_name=ONLINE_STORE),
        trigger=TableTrigger())

offline_tables = sorted({m.table_name for f in features
                         for m in fe.list_materialized_features(feature_name=f.full_name) if not m.is_online})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Populated offline table
# MAGIC On the first materialization the backfill is asynchronous (~10-20 min): waits until the offline table has data
# MAGIC and fails the task if that does not happen within the limit (the job retry reruns idempotently).

# COMMAND ----------

def populated(table):
    try:
        return spark.table(table).limit(1).count() > 0
    except Exception:  # table not created by the pipeline yet
        return False

deadline = time.time() + 30 * 60
while not all(populated(t) for t in offline_tables):
    if time.time() > deadline:
        raise TimeoutError(f"Offline table without data after 30 min: {offline_tables}")
    time.sleep(30)
