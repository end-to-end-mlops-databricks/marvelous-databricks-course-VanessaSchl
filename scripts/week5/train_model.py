"""This script trains a XGBoost Classifier model for predicting cancelation on hotel reservation."""

from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import argparse
import mlflow
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.reservations_model import ReservationsModel


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()

config_path = f"{args.root_path}/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Load training and test sets
train_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.train_set_vs"
).toPandas()
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set_vs"
).toPandas()

# Split features and target
y_train = train_set[[config.target]]
X_train = train_set.drop(columns=[config.target])
y_test = test_set[[config.target]]
X_test = test_set.drop(columns=[config.target])

# Setup preprocessing and model pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", DataProcessor(config=config)),
        ("classifier", ReservationsModel(config=config)),
    ]
)

mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-basic")

with mlflow.start_run(
    tags={
        "branch": "week5",
        "git_sha": f"{args.git_sha}",
        "job_run_id": args.job_run_id,
    }
) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"AUC: {auc}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "XGBoost Classifier with preprocessing")
    mlflow.log_params(config.parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("auc", auc)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    train_set_spark = spark.table(
        f"{config.catalog_name}.{config.schema_name}.train_set_vs"
    )
    dataset = mlflow.data.from_spark(
        train_set_spark,
        table_name=f"{config.catalog_name}.{config.schema_name}.train_set_vs",
        version="0",
    )
    mlflow.log_input(dataset, context="training")

    # Log model
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "code/hotel_reservations-3.0.0-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        code_paths=["../hotel_reservations-3.0.0-py3-none-any.whl"],
        artifact_path="vs-svc-pipeline-model",
        signature=signature,
        conda_env=conda_env,
    )

model_uri = f"runs:/{run_id}/vs-svc-pipeline-model"
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)
