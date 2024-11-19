# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-2.0.3-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from pyspark.sql import SparkSession

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
data_processor = DataProcessor(config=config, spark=spark)
train_set, test_set = data_processor.split_data(X=df)
data_processor.save_to_catalog(train_set=train_set, test_set=test_set)
