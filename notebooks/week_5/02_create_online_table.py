# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-3.0.0-py3-none-any.whl --force-reinstall

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
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

# Load config
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Define table names
feature_table_name = f"{config.catalog_name}.{config.schema_name}.hotel_features"
online_table_name = (
    f"{config.catalog_name}.{config.schema_name}.hotel_features_online"
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
            "no_of_previous_cancellations",
            "avg_price_per_room",
        ],
    )
]
