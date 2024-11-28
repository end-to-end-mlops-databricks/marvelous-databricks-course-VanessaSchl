"""This script handles data ingestion and feature table updates for a house price prediction system."""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max
from databricks.sdk import WorkspaceClient
import time
from hotel_reservations.config import ProjectConfig

workspace = WorkspaceClient()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
config_path = f"{args.root_path}/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path)

spark = SparkSession.builder.getOrCreate()

# Load source_data table
source_data = spark.table(f"{config.catalog_name}.{config.schema_name}.source_data_vs")

# Get max update timestamps from existing data
max_train_timestamp = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.train_set_vs")
    .select(spark_max("update_timestamp_utc").alias("max_update_timestamp"))
    .collect()[0]["max_update_timestamp"]
)

max_test_timestamp = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.test_set_vs")
    .select(spark_max("update_timestamp_utc").alias("max_update_timestamp"))
    .collect()[0]["max_update_timestamp"]
)

latest_timestamp = max(max_train_timestamp, max_test_timestamp)

# Filter source_data for rows with update_timestamp_utc greater than the latest_timestamp
new_data = source_data.filter(col("update_timestamp_utc") > latest_timestamp)

# Split the new data into train and test sets
new_data_train, new_data_test = new_data.randomSplit([0.8, 0.2], seed=42)

# Update train_set and test_set tables
new_data_train.write.mode("append").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.train_set_vs"
)
new_data_test.write.mode("append").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.test_set_vs"
)

# Verify affected rows count for train and test
affected_rows_train = new_data_train.count()
affected_rows_test = new_data_test.count()

# write into feature table; update online table
if affected_rows_train > 0 or affected_rows_test > 0:
    spark.sql(
        f"""
        WITH max_timestamp AS (
            SELECT MAX(update_timestamp_utc) AS max_update_timestamp
            FROM {config.catalog_name}.{config.schema_name}.train_set_vs
        )
        INSERT INTO {config.catalog_name}.{config.schema_name}.hotel_features
        SELECT Booking_ID, no_of_previous_cancellations, avg_price_per_room
        FROM {config.catalog_name}.{config.schema_name}.train_set_vs
        WHERE update_timestamp_utc == (SELECT max_update_timestamp FROM max_timestamp)
"""
    )
    spark.sql(
        f"""
        WITH max_timestamp AS (
            SELECT MAX(update_timestamp_utc) AS max_update_timestamp
            FROM {config.catalog_name}.{config.schema_name}.test_set_vs
        )
        INSERT INTO {config.catalog_name}.{config.schema_name}.hotel_features
        SELECT Booking_ID, no_of_previous_cancellations, avg_price_per_room
        FROM { config.catalog_name}.{config.schema_name}.test_set_vs
        WHERE update_timestamp_utc == (SELECT max_update_timestamp FROM max_timestamp)
"""
    )
    refreshed = 1
    update_response = workspace.pipelines.start_update(
        pipeline_id=config.pipeline_id, full_refresh=False
    )
    while True:
        update_info = workspace.pipelines.get_update(
            pipeline_id=config.pipeline_id, update_id=update_response.update_id
        )
        state = update_info.update.state.value
        if state == "COMPLETED":
            break
        elif state in ["FAILED", "CANCELED"]:
            raise SystemError("Online table failed to update.")
        elif state == "WAITING_FOR_RESOURCES":
            print("Pipeline is waiting for resources.")
        else:
            print(f"Pipeline is in {state} state.")
        time.sleep(30)
else:
    refreshed = 0

dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)
