# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Training and tuning of the fraud model
# MAGIC
# MAGIC Task `train` of the `fraud_training` job (depends on `feature_batch` and `feature_ondemand`):
# MAGIC
# MAGIC 1. Builds the declarative **training set** over `raw_fraud_labels`, resolving the batch and
# MAGIC    on-demand Feature Views point-in-time.
# MAGIC 2. Runs **tuning** with HyperOpt — one parent run and one nested run per evaluation in MLflow.
# MAGIC 3. Trains the **final model** and logs it with the **feature spec** (`fe.log_model`), **without registering it in UC**
# MAGIC    — registration belongs to the `register` task, gated by `quality_gate`.
# MAGIC 4. Publishes `test_auc` and `model_uri` as **task values** for the downstream tasks.
# MAGIC
# MAGIC All runs receive the MLflow tags `team` and `retention`, read from the job parameters.
# MAGIC Dependencies through the `train` serverless environment (`databricks-feature-engineering>=0.16.0`, `hyperopt`,
# MAGIC `xgboost`). Rerunnable: each run creates new runs, without side effects in UC.

# COMMAND ----------

# DBTITLE 1,Job parameters
# Job parameters arrive as widgets; the defaults apply to interactive runs
dbutils.widgets.text("team", "fraud")
dbutils.widgets.text("retention", "365d")
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("experiment", "")  # MLflow experiment of the target

TAGS = {"team": dbutils.widgets.get("team"), "retention": dbutils.widgets.get("retention")}

CATALOG      = dbutils.widgets.get("catalog")
SCHEMA       = dbutils.widgets.get("schema")
LABELS_TABLE = "raw_fraud_labels"  # spine: holder_id, created_at_brt, amount, label
EXPERIMENT   = dbutils.widgets.get("experiment")

def fqn(name): return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Declarative training set
# MAGIC `fe.create_training_set` computes each feature only with data before the `created_at_brt` of
# MAGIC each row (no leakage). `holder_id` is a lookup key and stays out of the model.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

BATCH_FEATURES    = [f"batch_pix_{m}_{w}d" for m in ("amount", "transfers") for w in (1, 7, 30)]
ONDEMAND_FEATURES = ["amount"]  # ondemand_day_of_week left out: limit of 5 UC UDF calls per training query
features = [fe.get_feature(full_name=fqn(n)) for n in BATCH_FEATURES + ONDEMAND_FEATURES]

training_set = fe.create_training_set(
    df=spark.table(fqn(LABELS_TABLE)),
    features=features,
    label="label",
    exclude_columns=["holder_id"])

training_df = training_set.load_df()
MODEL_COLS = [c for c in training_df.columns if c != "label"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Preparation and train/validation/test split
# MAGIC All features are numeric; `created_at_brt` is dropped inside the sklearn Pipeline (same
# MAGIC code in training and serving).

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pdf = training_df.toPandas()
pdf[MODEL_COLS] = pdf[MODEL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

X = pdf[MODEL_COLS]
y = pdf["label"].astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tuning with HyperOpt
# MAGIC Each evaluation becomes a **nested run** under the parent run `hyperopt_tuning`; the loss is `-val_auc`.
# MAGIC Sequential `Trials()` (serverless has no `SparkTrials`). The `team`/`retention` tags are passed
# MAGIC in `mlflow.start_run(tags=...)` of each run.

# COMMAND ----------

import mlflow
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlflow.set_experiment(EXPERIMENT)

def make_pipeline(params, n_estimators):
    # created_at_brt is only the point-in-time anchor
    pre = ColumnTransformer([("drop", "drop", ["created_at_brt"])], remainder="passthrough")
    return Pipeline([("pre", pre),
                     ("clf", XGBClassifier(n_estimators=n_estimators, tree_method="hist",
                                           eval_metric="logloss", **params))])

scale = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))  # class imbalance
search_space = {
    "max_depth":        hp.quniform("max_depth", 2, 8, 1),
    "learning_rate":    hp.loguniform("learning_rate", np.log(0.02), np.log(0.3)),
    "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
    "subsample":        hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "scale_pos_weight": hp.uniform("scale_pos_weight", scale * 0.8, scale * 1.2),
}

def to_int(params):
    return {**params, "max_depth": int(params["max_depth"]), "min_child_weight": int(params["min_child_weight"])}

def evaluate(params):
    params = to_int(params)
    with mlflow.start_run(nested=True, tags=TAGS):
        model = make_pipeline(params, n_estimators=200)
        model.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        mlflow.log_params(params)
        mlflow.log_metric("train_auc", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
        mlflow.log_metric("val_auc", val_auc)
    return {"loss": -val_auc, "status": STATUS_OK}

MAX_EVALS = 10
with mlflow.start_run(run_name="hyperopt_tuning", tags=TAGS):
    trials = Trials()
    argmin = fmin(fn=evaluate, space=search_space, algo=tpe.suggest, max_evals=MAX_EVALS,
                  trials=trials, rstate=np.random.default_rng(42))
    mlflow.log_metric("best_val_auc", max(-t["result"]["loss"] for t in trials.trials))

best_params = to_int(space_eval(search_space, argmin))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Final model logged with the feature spec (no UC registration)
# MAGIC `fe.log_model(training_set=...)` persists the **feature spec** (batch and on-demand features)
# MAGIC with the model. Without `registered_model_name`, no version is created in UC.

# COMMAND ----------

with mlflow.start_run(run_name="final_model", tags=TAGS) as run:
    final = make_pipeline(best_params, n_estimators=300)
    final.fit(X_train, y_train)
    test_auc = float(roc_auc_score(y_test, final.predict_proba(X_test)[:, 1]))
    mlflow.log_params(best_params)
    mlflow.log_metric("test_auc", test_auc)
    fe.log_model(
        model=final,
        artifact_path="model",
        flavor=mlflow.sklearn,      # Pipeline (encoder + XGBoost) → serving == training
        training_set=training_set,
        infer_input_example=True)   # signature inferred from the spine

model_uri = f"runs:/{run.info.run_id}/model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Task values for the downstream tasks
# MAGIC `quality_gate` compares `{{tasks.train.values.test_auc}}` with `min_auc`; `register` reads `model_uri`
# MAGIC through `dbutils.jobs.taskValues.get(taskKey="train", key="model_uri")`.

# COMMAND ----------

dbutils.jobs.taskValues.set(key="test_auc", value=test_auc)
dbutils.jobs.taskValues.set(key="model_uri", value=model_uri)
