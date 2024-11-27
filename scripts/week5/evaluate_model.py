"""This script evaluates and compares a new hotel reservations model against the currently deployed model."""

import argparse
from datetime import datetime
from databricks.sdk import WorkspaceClient
import mlflow
import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame, SparkSession

from hotel_reservations.config import ProjectConfig


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--new_model_uri",
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

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()

config_path = f"{args.root_path}/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path)

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Define the serving endpoint
serving_endpoint_name = "hotel-reservations-model-serving-vs"
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"

# Load test set and create additional features in Spark DataFrame
current_year = datetime.now().year
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set_vs"
).toPandas()

# Select the necessary columns for prediction and target
y_test = test_set[[config.target]]
X_test = test_set.drop(columns=[config.target])

# Generate predictions from both models
predictions_old = mlflow.sklearn.load_model(model_uri=previous_model_uri).predict(
    X=X_test
)
predictions_new = mlflow.sklearn.load_model(model_uri=args.new_model_uri).predict(
    X=X_test
)

test_set = test_set[["Booking_ID", "booking_status"]]

# Concatenate predictions with test set
spark_df = DataFrame(
    pd.concat(
        [
            test_set,
            pd.DataFrame(predictions_old, columns=["prediction_old"]),
            pd.DataFrame(predictions_new, columns=["prediction_new"]),
        ],
        axis=1,
        ignore_index=True,
    )
)

# Calculate the Area under ROC curve (AUC) for each model
evaluator = BinaryClassificationEvaluator(
    labelCol="booking_status", predictionCol="prediction_new", metricName="areaUnderROC"
)
auc_new = evaluator.evaluate(spark_df)

evaluator.setPredictionCol("prediction_old")
auc_old = evaluator.evaluate(spark_df)

# Compare models based on MAE and RMSE
print(f"AUC for New Model: {auc_new}")
print(f"AUC for Old Model: {auc_old}")

if auc_new < auc_old:
    print("New model is better based on AUC.")
    model_version = mlflow.register_model(
        model_uri=args.new_model_uri,
        name=f"{config.catalog_name}.{config.schema_name}.vs_hotel_reservations_model_basic",
        tags={"git_sha": f"{args.git_sha}", "job_run_id": args.job_run_id},
    )

    print("New model registered with version:", model_version.version)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    dbutils.jobs.taskValues.set(key="model_update", value=1)
else:
    print("Old model is better based on AUC.")
    dbutils.jobs.taskValues.set(key="model_update", value=0)
