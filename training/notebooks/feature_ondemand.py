# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Feature engineering / On-demand
# MAGIC
# MAGIC Task `feature_ondemand` of the `fraud_training` job. Ensures, **idempotently**, the
# MAGIC **on-demand features** — computed at training/inference time from the request payload
# MAGIC (`RequestSource`), without materialization:
# MAGIC
# MAGIC | Feature | Function | Payload input |
# MAGIC |---|---|---|
# MAGIC | `amount` | `ColumnSelection` — exposes the field as a feature | `amount` (DOUBLE) |
# MAGIC | `ondemand_day_of_week` | `CustomUDF` — UC UDF `day_of_week` | `created_at_brt` (TIMESTAMP) |
# MAGIC
# MAGIC Features already registered are reused (`get_feature`). Independent of the `feature_batch` task
# MAGIC (both run in parallel). Dependencies through the serverless environment of the task
# MAGIC (`databricks-feature-engineering>=0.16.0`).

# COMMAND ----------

# DBTITLE 1,Parameters
# Job parameters arrive as widgets (values of the bundle target)
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")

def fqn(name): return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities import (
    RequestSource, CustomUDF, ColumnSelection, FieldDefinition, ScalarDataType)

fe = FeatureEngineeringClient()

def ensure_feature(name, **kw):
    try:
        return fe.get_feature(full_name=fqn(name))  # already registered: reuse
    except Exception:
        return fe.create_feature(catalog_name=CATALOG, schema_name=SCHEMA, name=name, **kw)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `amount` — payload field as a feature
# MAGIC With `ColumnSelection` over `RequestSource`, the **feature name must match the column name**.

# COMMAND ----------

ensure_feature(
    "amount",
    source=RequestSource(schema=[FieldDefinition(name="amount", data_type=ScalarDataType.DOUBLE)]),
    function=ColumnSelection("amount"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `ondemand_day_of_week` — UC UDF over the payload timestamp
# MAGIC The UC Python function `day_of_week` is registered if it does not exist yet (`CREATE FUNCTION IF NOT EXISTS`).
# MAGIC `input_bindings` binds the UDF parameter (`dt_transaction`) to the `created_at_brt` field of the request.

# COMMAND ----------

spark.sql(f"""
CREATE FUNCTION IF NOT EXISTS {CATALOG}.{SCHEMA}.day_of_week(dt_transaction TIMESTAMP)
RETURNS STRING LANGUAGE PYTHON
COMMENT 'Returns the full English day-of-week name for a given transaction timestamp.'
AS $$
return dt_transaction.strftime("%A")
$$
""")

# COMMAND ----------

ensure_feature(
    "ondemand_day_of_week",
    source=RequestSource(schema=[FieldDefinition(name="created_at_brt", data_type=ScalarDataType.TIMESTAMP)]),
    function=CustomUDF(
        function_name=fqn("day_of_week"),
        input_bindings={"dt_transaction": "created_at_brt"}))
