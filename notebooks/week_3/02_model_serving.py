# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-2.2.2-py3-none-any.whl --force-reinstall

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import time

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()

# COMMAND ----------
# Load config
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Create online table of hotel features
online_table_name = f"{catalog_name}.{schema_name}.hotel_features_online"
spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=f"{catalog_name}.{schema_name}.hotel_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
        {"triggered": "true"}
    ),
    perform_full_copy=False,
)

online_table_pipeline = workspace.online_tables.create(
    name=online_table_name, spec=spec
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create endpoint

# COMMAND ----------
workspace.serving_endpoints.create(
    name="hotel-reservations-model-serving-fe-vs",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.vs_hotel_reservations_model_fe",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=4,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the endpoint

# COMMAND ----------
token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------
# Excluding "no_of_previous_cancellations" and "avg_price_per_room"
# because they will be taken from feature look up
required_columns = [
    "Booking_ID",
    "no_of_adults",
    "no_of_children",
    "type_of_meal_plan",
    "required_car_parking_space",
    "room_type_reserved",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "market_segment_type",
    "repeated_guest",
    "no_of_previous_bookings_not_canceled",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "no_of_special_requests",
]

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_vs").toPandas()

sampled_records = (
    train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
)
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
train_set.dtypes

# COMMAND ----------
dataframe_records[0]

# COMMAND ----------
start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-model-serving-fe-vs/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")
