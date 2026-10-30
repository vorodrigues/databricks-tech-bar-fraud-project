# fraud_mlops — MLOps bundle of the PIX fraud model

Databricks Asset Bundle (DABs) in the [MLOps Stacks](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-stacks) layout that packages the lifecycle of the PIX transaction fraud detection model — data, features, training, registration, batch scoring and online serving — and promotes it across environments with GitHub Actions.

## Layout

```
databricks.yml                              # bundle, variables and targets
resources/
  ml-artifacts-resource.yml                 # MLflow experiment and Unity Catalog registered model
  setup-workflow-resource.yml               # job fraud_setup
  ingest-pipeline-resource.yml              # landing volume and ingestion pipeline (Lakeflow SDP)
  training-workflow-resource.yml            # job fraud_training
  batch-scoring-workflow-resource.yml       # job fraud_batch_scoring
  online-deploy-workflow-resource.yml       # job fraud_online_deploy
setup/notebooks/                            # data, online_store, tag_policy
setup/pipelines/                            # ingest (bronze and silver tables)
training/notebooks/                         # feature_batch, feature_ondemand, train, register
scoring/notebooks/                          # batch_scoring
serving/notebooks/                          # online_deploy
.github/workflows/
  ci.yml                                    # PR → validate, deploy and training in staging
  cd.yml                                    # merge to main → validate and deploy in prod
.DRAFTS/                                    # one-off tests (outside git and the deploy)
```

## Targets

All targets use the same workspace and Serverless compute.

| Target | Mode | Deploy and `run_as` | UC schema | Root directory |
|---|---|---|---|---|
| `dev` (default) | `development` | developer user | `vr_workspace.tech_bar_dev` | `/Workspace/Users/victor.rodrigues@databricks.com/tech_bar/<project>/dev` |
| `staging` | `production` | SP `cicd-ml-serving` (via CI) | `vr_workspace.tech_bar_staging` | `/Workspace/Users/victor.rodrigues@databricks.com/tech_bar/<project>/staging` |
| `prod` | `production` | SP `cicd-ml-serving` (via CD) | `vr_workspace.tech_bar` | `/Workspace/Users/victor.rodrigues@databricks.com/tech_bar/<project>/prod` |

In `development`, DABs prefixes resource names with the user (e.g. experiment `[dev <user>] fraud_experiment`, model `dev_<user>_fraud_model`) and pauses schedules, isolating the work of each developer.

## Variables

| Variable | Default | Usage |
|---|---|---|
| `catalog` | `vr_workspace` | UC catalog |
| `schema` | per target | UC schema of the target |
| `prefix` | `${bundle.target}` | prefix of assets without a directory (jobs, endpoint) |
| `online_store` | `feature-store-online` | Lakebase project shared across targets |
| `service_principal` | SP application id | `run_as` of `staging`/`prod` |
| `viewers_group` | `ml-viewers` | group with run/view permission |

The code is deployed to `${workspace.file_path}` (`<root directory>/files`). The jobs receive these values as job parameters and the notebooks read them through `dbutils.widgets` — no environment value is hardcoded.

## Resources

| Resource | Name | Description |
|---|---|---|
| MLflow experiment | `${workspace.root_path}/fraud_experiment` | tuning and final training runs |
| Registered model (UC) | `${var.catalog}.${var.schema}.fraud_model` | versions with the `champion` alias |
| UC Volume | `${var.catalog}.${var.schema}.landing` | synthetic source files (JSON) in `pix_transactions/` and `fraud_labels/` |
| Pipeline (Lakeflow SDP) | `${var.prefix}_fraud_ingest` | Serverless, triggered: bronze streaming tables (`bronze_pix_transactions`, `bronze_fraud_labels`) loaded by Auto Loader from the volume; silver materialized views (`raw_pix_transactions`, `raw_fraud_labels`) with expectations that drop invalid rows |
| Job | `${var.prefix}_fraud_setup` | `data` (writes the volume files) → `ingest` (pipeline update); `online_store` and `tag_policy` independent; all idempotent |
| Job | `${var.prefix}_fraud_training` | `feature_batch` + `feature_ondemand` → `train` → `quality_gate` (`test_auc >= min_auc`) → `register` |
| Job | `${var.prefix}_fraud_batch_scoring` | scores the source table with `fraud_model@champion` (daily at 06:00 BRT, paused) |
| Job | `${var.prefix}_fraud_online_deploy` | creates/updates the route-optimized endpoint `${var.prefix}_fraud_serving` with an inference table |

All jobs grant `CAN_MANAGE_RUN` to the `ml-viewers` group.

## Prerequisites

- Service principal `cicd-ml-serving` and group `ml-viewers` in the workspace.
- Folder `/Workspace/Users/victor.rodrigues@databricks.com/tech_bar` with `CAN_MANAGE` for the SP.
- Schemas `vr_workspace.tech_bar_dev`, `vr_workspace.tech_bar_staging` and `vr_workspace.tech_bar`, with privileges for the user and the SP.
- Secret scope `mlops` (`sp-client-id`, `sp-client-secret`) for the endpoint tests.
- GitHub repository secrets: `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` (SP OAuth M2M).
- A single run of `fraud_setup` in each target before the other jobs.

## Development in `dev`

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev fraud_setup        # once
databricks bundle run -t dev fraud_training
databricks bundle run -t dev fraud_batch_scoring
databricks bundle run -t dev fraud_online_deploy
```

## Promotion flow

1. **dev** — development and tests with the developer's own user.
2. **staging** — a PR to `main` triggers the CI (`ci.yml`): `bundle validate`, `bundle deploy` and `bundle run fraud_training` in `staging`, as the SP. The workflow waits for the run and fails if it fails.
3. **prod** — the merge to `main` triggers the CD (`cd.yml`): `bundle validate` and `bundle deploy` in `prod`, as the SP, without running jobs.

Each workflow uses one `concurrency` group per target, without cancelling runs in progress. Both can also be triggered manually (`workflow_dispatch`). `staging` and `prod` are changed only by the CI/CD pipeline.
