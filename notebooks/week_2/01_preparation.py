# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-3.0.0-py3-none-any.whl --force-reinstall

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# Load the house prices dataset
df = spark.read.csv(
    "/Volumes/dev/datalab_1ai/files/hotel_reservations.csv",
    header=True,
    inferSchema=True,
).toPandas()

# COMMAND ----------
data_processor = DataProcessor(config=config)
train_set, test_set = data_processor.split_data(X=df)

# Save the train and test sets into Databricks tables
train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

train_set_with_timestamp.write.mode("append").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.train_set_vs"
)

test_set_with_timestamp.write.mode("append").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.test_set_vs"
)

spark.sql(
    f"ALTER TABLE {config.catalog_name}.{config.schema_name}.train_set_vs "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

spark.sql(
    f"ALTER TABLE {config.catalog_name}.{config.schema_name}.test_set_vs "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)
