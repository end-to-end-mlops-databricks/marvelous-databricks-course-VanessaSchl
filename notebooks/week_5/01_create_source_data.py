# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-3.0.0-py3-none-any.whl --force-reinstall

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from hotel_reservations.config import ProjectConfig

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# Load train and test sets
train_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.train_set_vs"
).toPandas()
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set_vs"
).toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)
existing_ids = set(str(id) for id in combined_set["Booking_ID"])


# COMMAND ----------
# Define function to create synthetic data without random state
def create_synthetic_data(df, num_rows=100, existing_ids=None):
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != "Booking_ID":
            if column == "arrival_year":
                synthetic_data[column] = np.random.randint(
                    2019, 2024, num_rows
                )  # Years after existing values
            elif column in ["arrival_month", "arrival_date"]:
                synthetic_data[column] = np.random.randint(
                    df[column].min(), df[column].max(), num_rows
                )
            elif column == "booking_status":
                synthetic_data[column] = np.random.randint(0, 2, num_rows)
            else:
                mean, std = df[column].mean(), df[column].std()
                synthetic_data[column] = np.random.normal(mean, std, num_rows)

        elif pd.api.types.is_categorical_dtype(
            df[column]
        ) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif isinstance(df[column].dtype, pd.CategoricalDtype) or isinstance(
            df[column].dtype, pd.StringDtype
        ):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(
                    np.random.randint(min_date.value, max_date.value, num_rows)
                )
            else:
                synthetic_data[column] = [min_date] * num_rows

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    new_ids = []
    i = len(existing_ids)
    while len(new_ids) < num_rows:
        new_ids.append("INN" + str(i))  # Convert numeric ID to string
        i += 1
    synthetic_data["Booking_ID"] = new_ids

    return synthetic_data


# COMMAND ----------
# Create synthetic data
synthetic_df = create_synthetic_data(
    combined_set, num_rows=10000, existing_ids=existing_ids
)

existing_schema = spark.table(
    f"{config.catalog_name}.{config.schema_name}.train_set_vs"
).schema

synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

# COMMAND ----------
# Append synthetic data as new data to source_data table
train_set_with_timestamp.write.mode("append").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.source_data_vs"
)
