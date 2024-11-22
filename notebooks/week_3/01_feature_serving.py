# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-2.2.0-py3-none-any.whl --force-reinstall

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import time

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
host = spark.conf.get("spark.databricks.workspaceUrl")

# Initialize Databricks clients
workspace = WorkspaceClient(host=host, token=token)
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config, train and test tables

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Get feature columns details
num_features = config.num_features
original_target = config.original_target
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features_preds"
online_table_name = f"{catalog_name}.{schema_name}.hotel_features_preds_online"

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_vs").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_vs").toPandas()

df = pd.concat([train_set, test_set])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a registered model

# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(
    f"models:/{catalog_name}.{schema_name}.vs_hotel_reservations_model_basic/1"
)

# COMMAND ----------

# Prepare the DataFrame for predictions and feature table creation - these features are the ones we want to serve.
preds_df = df[
    [
        "Booking_ID",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "no_of_previous_cancellations",
        "avg_price_per_room",
    ]
]
preds_df["predicted_booking_status_canceled"] = pipeline.predict(
    df.drop(columns=[original_target])
)

preds_df = spark.createDataFrame(preds_df)

# 1. Create the feature table in Databricks
fe.create_table(
    name=feature_table_name,
    primary_keys=["Booking_ID"],
    df=preds_df,
    description="Hotel Reservations predictions feature table",
)

# Enable Change Data Feed
spark.sql(
    f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""
)

# COMMAND ----------
# 2. Create the online table using feature table
spec = OnlineTableSpec(
    primary_key_columns=["Booking_ID"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
        {"triggered": "true"}
    ),
    perform_full_copy=False,
)

# Create the online table in Databricks
online_table_pipeline = workspace.online_tables.create(
    name=online_table_name, spec=spec
)

# COMMAND ----------
# 3. Create feture look up and feature spec table feature table

# Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name,
        lookup_key="Booking_ID",
        feature_names=[
            "no_of_weekend_nights",
            "no_of_week_nights",
            "no_of_previous_cancellations",
            "avg_price_per_room",
            "predicted_booking_status_canceled",
        ],
    )
]

# Create the feature spec for serving
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions_vs"

fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Feature Serving Endpoint

# COMMAND ----------
# 4. Create endpoing using feature spec
# Create a serving endpoint for the house prices predictions
workspace.serving_endpoints.create(
    name="hotel-reservations-feature-serving-vs",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,  # feature spec name defined in the previous step
                scale_to_zero_enabled=True,
                workload_size="Small",  # Define the workload size (Small, Medium, Large)
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call The Endpoint

# COMMAND ----------

id_list = preds_df["Booking_ID"]

# COMMAND ----------
start_time = time.time()

serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-feature-serving-vs/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"Booking_ID": "INN30213"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")
