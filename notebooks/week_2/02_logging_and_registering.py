# Databricks notebook source
# MAGIC %pip install ../housing_price-1.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

import pandas as pd
from pyspark.sql import SparkSession
from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.reservations_model import ReservationsModel
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri(
    "databricks-uc"
)  # It must be -uc for registering models to Unity Catalog

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details
num_features = config["num_features"]
target = config["target"]
parameters = config["parameters"]
catalog_name = config["catalog_name"]
schema_name = config["schema_name"]

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
pandas_df = pd.read_csv("/Volumes/dev/datalab_1ai/files/hotel_reservations.csv")
preprocessor = DataProcessor(config=config, spark=spark)
train_set, test_set = preprocessor.split_data(pandas_df)
preprocessor.save_to_catalog(train_set, test_set)

X_train = train_set[num_features].values
y_train = train_set[target].values

X_test = test_set[num_features].values
y_test = test_set[target].values

# COMMAND ----------

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(
    steps=[
        ("preprocessor", DataProcessor(config=config, spark=spark)),
        ("classifier", ReservationsModel(config=config)),
    ]
)


# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations")
GIT_SHA = "ffa63b430205ff7"

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{GIT_SHA}", "branch": "week1+2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    accuracy, precision = pipeline.named_steps["classifier"].evaluate(X_test, y_test)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "SVC with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set_vs")
    dataset = mlflow.data.from_spark(
        train_set_spark,
        table_name=f"{catalog_name}.{schema_name}.train_set_vs",
        version="0",
    )
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(
        sk_model=pipeline, artifact_path="vs-svc-pipeline-model", signature=signature
    )


# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/vs-svc-pipeline-model",
    name=f"{catalog_name}.{schema_name}.vs_hotel_reservations_model_basic",
    tags={"git_sha": f"{GIT_SHA}"},
)

# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()
