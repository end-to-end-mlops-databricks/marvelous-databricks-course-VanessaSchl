# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-3.0.0-py3-none-any.whl --force-reinstall

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.reservations_model import ReservationsModel

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Load training and testing sets from Databricks tables
train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set_vs").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set_vs").toPandas()

y_train = train_set[[config.target]]
X_train = train_set.drop(columns=[config.target, "update_timestamp_utc"])

y_test = test_set[[config.target]]
X_test = test_set.drop(columns=[config.target, "update_timestamp_utc"])

# COMMAND ----------
# Create the pipeline with preprocessing and SVC
pipeline = Pipeline(
    steps=[
        ("preprocessor", DataProcessor(config=config)),
        ("classifier", ReservationsModel(config=config)),
    ]
)


# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-vs")
GIT_SHA = "ffa63b430205ff7"

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{GIT_SHA}", "branch": "week1+2"},
) as run:
    run_id = run.info.run_id

    y_train = y_train.replace({"Not_Canceled": "0", "Canceled": "1"}).astype(int).values
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_test = y_test.replace({"Not_Canceled": "0", "Canceled": "1"}).astype(int).values

    # Evaluate the model performance
    accuracy, precision = pipeline.named_steps["classifier"].evaluate(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "SVC with preprocessing")
    mlflow.log_params(config.parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    train_set_spark = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set_vs")
    dataset = mlflow.data.from_spark(
        train_set_spark,
        table_name=f"{config.catalog_name}.{config.schema_name}.train_set_vs",
        version="0",
    )
    mlflow.log_input(dataset, context="training")

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

# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/vs-svc-pipeline-model",
    name=f"{config.catalog_name}.{config.schema_name}.vs_hotel_reservations_model_basic",
    tags={"git_sha": f"{GIT_SHA}"},
)

# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()
