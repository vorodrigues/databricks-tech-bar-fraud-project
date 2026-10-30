# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — governed tags `confidentiality` and `team`
# MAGIC
# MAGIC **Governed tags** are tags with a *tag policy* defined at account level: the policy fixes the key and the list of allowed values, and only users with the `ASSIGN` permission can apply it to Unity Catalog objects.
# MAGIC
# MAGIC The policies are **shared** by all targets. This notebook ensures:
# MAGIC - `confidentiality` with the values `public`, `internal`, `confidential` and `restricted`;
# MAGIC - `team` containing the value `fraud`;
# MAGIC - the `ASSIGN` permission for the run identities of all targets.
# MAGIC
# MAGIC The tags are later applied to the registered model by the `register` task of the training job.
# MAGIC
# MAGIC Reference: [Governed tags](https://docs.databricks.com/aws/en/admin/governed-tags/)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Desired definition

# COMMAND ----------

dbutils.widgets.text("principals", "")  # comma-separated identities (user e-mail or SP application id)
PRINCIPALS = [p.strip() for p in dbutils.widgets.get("principals").split(",") if p.strip()]

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.iam import GrantRule, RuleSetUpdateRequest
from databricks.sdk.service.tags import TagPolicy, Value

POLICIES = {
    "confidentiality": ("Confidentiality classification of the asset", ["public", "internal", "confidential", "restricted"]),
    "team": ("Team that owns the asset", ["fraud"]),
}

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Idempotent creation
# MAGIC
# MAGIC SDK `tag_policies` API:
# MAGIC - missing policy → `create_tag_policy`;
# MAGIC - existing policy missing some value → `update_tag_policy` with `update_mask="values"`, adding only the missing values (existing ones are kept);
# MAGIC - compliant policy → no change.

# COMMAND ----------

actions = {}
for key, (description, values) in POLICIES.items():
    try:
        current = w.tag_policies.get_tag_policy(key)
    except NotFound:
        current = None

    if current is None:
        w.tag_policies.create_tag_policy(TagPolicy(tag_key=key, description=description, values=[Value(name=v) for v in values]))
        actions[key] = "created"
        continue

    existing = [v.name for v in current.values or []]
    missing = [v for v in values if v not in existing]
    if missing:
        w.tag_policies.update_tag_policy(
            key, TagPolicy(tag_key=key, values=[Value(name=v) for v in existing + missing]), update_mask="values")
        actions[key] = f"added {missing}"
    else:
        actions[key] = "unchanged"

# COMMAND ----------

# MAGIC %md
# MAGIC ## `ASSIGN` permission
# MAGIC
# MAGIC The permissions of each policy live in the account *rule set* `accounts/<account_id>/tagPolicies/<policy_id>/ruleSets/default` (Account Access Control Proxy API).
# MAGIC The role `roles/tagPolicy.assigner` is granted only to identities that have neither `assigner` nor `manager`, keeping the existing rules.

# COMMAND ----------

ACCOUNT_ID = w.api_client.do("GET", "/.well-known/databricks-config")["account_id"]
ASSIGNER, MANAGER = "roles/tagPolicy.assigner", "roles/tagPolicy.manager"
principal_ids = [f"users/{p}" if "@" in p else f"servicePrincipals/{p}" for p in PRINCIPALS]

for key in POLICIES:
    policy = w.tag_policies.get_tag_policy(key)
    name = f"accounts/{ACCOUNT_ID}/tagPolicies/{policy.id}/ruleSets/default"
    rule_set = w.account_access_control_proxy.get_rule_set(name, etag="")
    rules = rule_set.grant_rules or []
    granted = {p for r in rules if r.role in (ASSIGNER, MANAGER) for p in r.principals or []}
    missing = [p for p in principal_ids if p not in granted]
    if missing:
        others = [r for r in rules if r.role != ASSIGNER]
        assigners = [p for r in rules if r.role == ASSIGNER for p in r.principals or []]
        w.account_access_control_proxy.update_rule_set(name, RuleSetUpdateRequest(
            name=name, etag=rule_set.etag, grant_rules=others + [GrantRule(role=ASSIGNER, principals=assigners + missing)]))
        actions[key] += f"; assign {missing}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

for key, (_, values) in POLICIES.items():
    current = [v.name for v in w.tag_policies.get_tag_policy(key).values]
    assert set(values) <= set(current), (key, current)
print(actions)
dbutils.notebook.exit(str(actions))
