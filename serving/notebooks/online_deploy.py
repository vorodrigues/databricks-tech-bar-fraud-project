# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Model Serving of the PIX fraud model (route-optimized + inference table)
# MAGIC
# MAGIC Task `deploy` of the `fraud_online_deploy` job: publishes the `champion` version of the target `fraud_model`
# MAGIC (registered with the **feature spec**) to a **Model Serving endpoint**. On every request, the endpoint
# MAGIC automatically resolves the **6 batch features** (online lookup in the shared Lakebase) and the
# MAGIC on-demand `amount` (through `RequestSource`) — the client sends only `holder_id` + `created_at_brt` + `amount`.
# MAGIC
# MAGIC ## What this notebook does (idempotent)
# MAGIC 1. **Deploy through the Databricks SDK** — `create_and_wait` the first time; `update_config_and_wait` +
# MAGIC    `put_ai_gateway` if the endpoint already exists (never recreates it destructively).
# MAGIC 2. **Route optimization** (`route_optimized=True`) — low-latency route; set **only at
# MAGIC    creation**. Requires querying through `endpoint_url` (data plane) with an OAuth token.
# MAGIC 3. **Inference table through AI Gateway** — `AiGatewayInferenceTableConfig` writes the payloads to the
# MAGIC    target catalog/schema (replaces the legacy `auto_capture_config`).
# MAGIC 4. **Query permission** — grants `CAN_QUERY` on the endpoint to the viewers group (additive).
# MAGIC
# MAGIC > `databricks-feature-lookup` is built into Model Serving — no need to package it.

# COMMAND ----------

# DBTITLE 1,Endpoint parameters
# values come from the job parameters (bundle); empty in standalone runs
dbutils.widgets.text("endpoint_name", "")   # endpoint name (<prefix>_fraud_serving)
dbutils.widgets.text("model_name", "")      # UC model packaged with the feature spec (catalog.schema.name)
dbutils.widgets.text("catalog", "")         # target catalog/schema: destination of the inference table
dbutils.widgets.text("schema", "")
dbutils.widgets.text("viewers_group", "")   # group that receives CAN_QUERY on the endpoint

CATALOG        = dbutils.widgets.get("catalog")
PROJECT_SCHEMA = dbutils.widgets.get("schema")
ENDPOINT_NAME  = dbutils.widgets.get("endpoint_name")
VIEWERS_GROUP  = dbutils.widgets.get("viewers_group")
SERVED_NAME    = "fraud-model"                     # name of the served entity in the endpoint
INFERENCE_TABLE_PREFIX = "fraud_serving_inference"   # AI Gateway -> <cat>.<schema>.<prefix>_payload
ALIAS          = "champion"

MODEL_FULL_NAME = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md ## 1. Resolve the `champion` version of the model
# MAGIC The alias is resolved to a version number: the endpoint serves a fixed version, and a new deploy
# MAGIC follows the alias after each retraining.

# COMMAND ----------

from mlflow.tracking import MlflowClient

mlflow_client = MlflowClient(registry_uri="databricks-uc")
model_version = mlflow_client.get_model_version_by_alias(MODEL_FULL_NAME, ALIAS).version
print(f"model={MODEL_FULL_NAME} | version {ALIAS}={model_version}")

# COMMAND ----------

# MAGIC %md ## 2. Endpoint config (served entity + traffic + AI Gateway inference table)
# MAGIC - **served entity**: UC model at the `champion` version, CPU/Small workload, scale-to-zero.
# MAGIC - **AI Gateway inference table**: writes the request/response payloads to the target UC
# MAGIC   (`<catalog>.<schema>.<prefix>_payload`).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route,
    ServingModelWorkloadType, AiGatewayConfig, AiGatewayInferenceTableConfig,
)

w = WorkspaceClient()

served_entity = ServedEntityInput(
    name=SERVED_NAME,
    entity_name=MODEL_FULL_NAME,
    entity_version=model_version,
    workload_type=ServingModelWorkloadType.CPU,
    workload_size="Small",
    scale_to_zero_enabled=True,
)
core_config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,
    served_entities=[served_entity],
    traffic_config=TrafficConfig(routes=[Route(served_model_name=SERVED_NAME, traffic_percentage=100)]),
)
ai_gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        enabled=True,
        catalog_name=CATALOG,
        schema_name=PROJECT_SCHEMA,
        table_name_prefix=INFERENCE_TABLE_PREFIX,
    )
)

# COMMAND ----------

# MAGIC %md ## 3. Idempotent deploy
# MAGIC `create_and_wait` (with `route_optimized` + `ai_gateway`) only the first time. If it already exists, updates the
# MAGIC served config (`update_config_and_wait`) and the inference table (`put_ai_gateway`) — without recreating it.
# MAGIC `route_optimized` can only be set at creation (it does not change through update).

# COMMAND ----------

from databricks.sdk.errors import ResourceDoesNotExist

try:
    existing = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"endpoint '{ENDPOINT_NAME}' exists (route_optimized={existing.route_optimized}) -> update")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
        traffic_config=core_config.traffic_config,
    )
    w.serving_endpoints.put_ai_gateway(
        name=ENDPOINT_NAME,
        inference_table_config=ai_gateway.inference_table_config,
    )
except ResourceDoesNotExist:
    print(f"endpoint '{ENDPOINT_NAME}' does not exist -> create (route_optimized=True)")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=core_config,
        route_optimized=True,
        ai_gateway=ai_gateway,
    )

ep = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"state={ep.state.ready} | config_update={ep.state.config_update} | route_optimized={ep.route_optimized}")
print(f"endpoint_url={ep.endpoint_url}")
print(f"ai_gateway.inference_table={ep.ai_gateway.inference_table_config if ep.ai_gateway else None}")

# COMMAND ----------

# MAGIC %md ## 4. Query permission for the viewers group
# MAGIC `update_permissions` **adds** `CAN_QUERY` to the group, keeping the other endpoint permissions
# MAGIC (`set_permissions` would replace the whole ACL).

# COMMAND ----------

from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel

acl = w.serving_endpoints.update_permissions(
    serving_endpoint_id=ep.id,
    access_control_list=[ServingEndpointAccessControlRequest(
        group_name=VIEWERS_GROUP, permission_level=ServingEndpointPermissionLevel.CAN_QUERY)])
for a in acl.access_control_list:
    print(a.user_name or a.group_name or a.service_principal_name, [p.permission_level.value for p in a.all_permissions])

# COMMAND ----------

# DBTITLE 1,Structured evidence
import json

ep = w.serving_endpoints.get(ENDPOINT_NAME)
itc = ep.ai_gateway.inference_table_config if ep.ai_gateway else None
dbutils.notebook.exit(json.dumps({
    "endpoint_name":       ENDPOINT_NAME,
    "endpoint_id":         ep.id,
    "state_ready":         str(ep.state.ready),
    "config_update":       str(ep.state.config_update),
    "route_optimized":     ep.route_optimized,
    "endpoint_url":        ep.endpoint_url,
    "model_full_name":     MODEL_FULL_NAME,
    "model_version":       model_version,
    "inference_table":     f"{CATALOG}.{PROJECT_SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload",
    "inference_enabled":   (itc.enabled if itc else None),
}))
