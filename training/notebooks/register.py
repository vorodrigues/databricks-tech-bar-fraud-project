# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - UC registration, tags and `champion` alias
# MAGIC
# MAGIC Task `register` of the `fraud_training` job (runs only on the `true` branch of `quality_gate`):
# MAGIC
# MAGIC 1. Reads the **task value** `model_uri` published by `train` and registers a new **version** of
# MAGIC    the target `fraud_model` in Unity Catalog.
# MAGIC 2. Applies the tags `team` and `retention` (job parameters) to the **version** and to the **registered model**.
# MAGIC 3. Applies the **governed tag** `confidentiality = confidential` to the registered model — the key has a
# MAGIC    tag policy (`fraud_setup` job), so UC validates the value against the allowed list.
# MAGIC 4. Points the `champion` alias to the new version (the serving endpoint is not changed here).
# MAGIC
# MAGIC On registered models and versions, UC tags are applied through the MLflow API
# MAGIC (`set_registered_model_tag` / `set_model_version_tag`) — there is no SQL `SET TAG` for models.
# MAGIC Rerunnable (Repair run): if the source run already has a registered version, it is reused.

# COMMAND ----------

# DBTITLE 1,Job parameters
# team/retention/model_name come from the job parameters; model_uri is only set in standalone runs
dbutils.widgets.text("team", "fraud")
dbutils.widgets.text("retention", "365d")
dbutils.widgets.text("model_uri", "")  # e.g. runs:/<run_id>/model; empty → task value of `train`
dbutils.widgets.text("model_name", "")  # target registered model (catalog.schema.name)

TAGS = {"team": dbutils.widgets.get("team"), "retention": dbutils.widgets.get("retention")}
GOVERNED_TAGS = {"confidentiality": "confidential"}

MODEL_NAME = dbutils.widgets.get("model_name")
ALIAS = "champion"

model_uri = dbutils.widgets.get("model_uri") or dbutils.jobs.taskValues.get(taskKey="train", key="model_uri")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. New version from the model logged by `train`
# MAGIC `mlflow.register_model` creates the version in UC, keeping the feature spec logged by `fe.log_model`.
# MAGIC For idempotency, a version already registered from the same run is reused.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

run_id = model_uri.split("/")[1]  # runs:/<run_id>/model
existing = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'") if v.run_id == run_id]
version = existing[0].version if existing else mlflow.register_model(model_uri, MODEL_NAME).version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. UC tags on the version and on the registered model
# MAGIC `team`/`retention` on the new version and on the model; `confidentiality` (governed) only on the model.

# COMMAND ----------

for key, value in TAGS.items():
    client.set_model_version_tag(MODEL_NAME, version, key, value)

for key, value in {**TAGS, **GOVERNED_TAGS}.items():
    client.set_registered_model_tag(MODEL_NAME, key, value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Promotion: `champion` alias → new version
# MAGIC Consumers of `fraud_model@champion` (e.g. the batch prediction job) start using this version.

# COMMAND ----------

client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
dbutils.notebook.exit(str(version))
