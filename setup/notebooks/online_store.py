# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — online feature store (Lakebase)
# MAGIC
# MAGIC The online store is a Lakebase project **shared** by all targets: the batch features are materialized in it and read by Model Serving at inference time.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. reuses the existing online store or creates it if missing (never recreates or removes it);
# MAGIC 2. ensures `CAN_USE` on the project for the run identities of all targets (required to materialize online features).
# MAGIC
# MAGIC Reference: [Online feature store](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store)

# COMMAND ----------

dbutils.widgets.text("online_store", "")
dbutils.widgets.text("principals", "")  # comma-separated identities (user e-mail or SP application id)

ONLINE_STORE = dbutils.widgets.get("online_store")
PRINCIPALS = [p.strip() for p in dbutils.widgets.get("principals").split(",") if p.strip()]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Online store
# MAGIC
# MAGIC `get_online_store` → `create_online_store` only if missing; waits for the `AVAILABLE` state.

# COMMAND ----------

import time
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

store = fe.get_online_store(name=ONLINE_STORE)
if store is None:
    store = fe.create_online_store(name=ONLINE_STORE, capacity="CU_1")
while "AVAILABLE" not in str(store.state):
    if any(s in str(store.state) for s in ("FAILED", "DELETING")):
        raise RuntimeError(f"Online store {ONLINE_STORE}: {store.state}")
    time.sleep(20)
    store = fe.get_online_store(name=ONLINE_STORE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Identity access
# MAGIC
# MAGIC Project permissions (`database-projects`): grants `CAN_USE` only to identities that do not have `CAN_USE` or `CAN_MANAGE` yet (additive update).
# MAGIC
# MAGIC Reading and changing the ACL requires `CAN_MANAGE`. An identity with only `CAN_USE` (which already accesses the project above) cannot check it: granting the other identities is left to whoever manages the project.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

w = WorkspaceClient()
OBJECT_TYPE = "database-projects"

try:
    acl = w.permissions.get(OBJECT_TYPE, ONLINE_STORE).access_control_list or []
except PermissionDenied:
    acl = None

if acl is None:
    missing = []
    print(f"No CAN_MANAGE on {ONLINE_STORE}: ACL not checked for {PRINCIPALS}")
else:
    granted = {
        a.user_name or a.service_principal_name
        for a in acl
        if any(p.permission_level in (PermissionLevel.CAN_USE, PermissionLevel.CAN_MANAGE) for p in a.all_permissions or [])
    }
    missing = [p for p in PRINCIPALS if p not in granted]
if missing:
    w.permissions.update(OBJECT_TYPE, ONLINE_STORE, access_control_list=[
        AccessControlRequest(user_name=p, permission_level=PermissionLevel.CAN_USE) if "@" in p
        else AccessControlRequest(service_principal_name=p, permission_level=PermissionLevel.CAN_USE)
        for p in missing
    ])

dbutils.notebook.exit(f"{store.name} {store.state} granted={missing} acl_checked={acl is not None}")
