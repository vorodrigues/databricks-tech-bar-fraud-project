# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Batch prediction with `fraud_model@champion`
# MAGIC
# MAGIC Task `score` of the `fraud_batch_scoring` job (scheduled daily, paused by default):
# MAGIC
# MAGIC 1. Reads the source table given by the `source_table` parameter (contract: `holder_id`, `created_at_brt`, `amount`).
# MAGIC 2. Resolves the version behind the `champion` alias and scores with `fe.score_batch` — the features
# MAGIC    (batch and on-demand Feature Views) are computed from
# MAGIC    the **feature spec** saved with the model,
# MAGIC    with point-in-time correctness, as in training.
# MAGIC 3. Writes the predictions to the target `fraud_batch_predictions` in **append** mode, with the model version,
# MAGIC    the source table and the scoring instant.

# COMMAND ----------

# DBTITLE 1,Job parameters
# source_table: fully qualified name (catalog.schema.table); the job parameter with the same name feeds the widget
dbutils.widgets.text("source_table", "")
dbutils.widgets.text("model_name", "")  # scored registered model (catalog.schema.name)
dbutils.widgets.text("catalog", "")  # target catalog/schema: destination of the predictions
dbutils.widgets.text("schema", "")
SOURCE_TABLE = dbutils.widgets.get("source_table")

MODEL_NAME = dbutils.widgets.get("model_name")
ALIAS = "champion"
OUTPUT_TABLE = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}.fraud_batch_predictions"
KEYS = ["holder_id", "created_at_brt", "amount"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Output table
# MAGIC Created once (`IF NOT EXISTS`); each run appends rows.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
  holder_id      STRING    COMMENT 'Transaction holder',
  created_at_brt TIMESTAMP COMMENT 'Transaction instant (BRT)',
  amount         DOUBLE    COMMENT 'Transaction amount',
  prediction     DOUBLE    COMMENT 'Model prediction (1 = fraud)',
  model_version  INT       COMMENT 'Version of fraud_model used (champion alias)',
  source_table   STRING    COMMENT 'Scored source table',
  scored_at      TIMESTAMP COMMENT 'Scoring instant'
)
COMMENT 'Batch predictions of fraud_model@champion (fraud_batch_scoring job)'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Scoring with `score_batch`
# MAGIC The alias is resolved to a version number before scoring: the version used is recorded
# MAGIC in the output even if `champion` changes during the run. Only the columns of the source contract are sent;
# MAGIC the remaining inputs come from the feature spec.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
version = int(MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version)

fe = FeatureEngineeringClient()
source_df = spark.table(SOURCE_TABLE).select(*KEYS)
scored = fe.score_batch(model_uri=f"models:/{MODEL_NAME}/{version}", df=source_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write (append)

# COMMAND ----------

(scored.select(
    *KEYS,
    F.col("prediction").cast("double").alias("prediction"),
    F.lit(version).alias("model_version"),
    F.lit(SOURCE_TABLE).alias("source_table"),
    F.current_timestamp().alias("scored_at"))
 .write.mode("append").saveAsTable(OUTPUT_TABLE))
