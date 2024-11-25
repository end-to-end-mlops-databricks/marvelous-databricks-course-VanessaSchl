# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-2.2.0-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from pyspark.sql import SparkSession

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
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
