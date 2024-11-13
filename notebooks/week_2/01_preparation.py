# Databricks notebook source
import pandas as pd
from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# Load the house prices dataset
df = spark.read.csv(
    "/Volumes/dev/datalab_1ai/files/hotel_reservations.csv", header=True, inferSchema=True
).toPandas()

# COMMAND ----------
data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess_data()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
