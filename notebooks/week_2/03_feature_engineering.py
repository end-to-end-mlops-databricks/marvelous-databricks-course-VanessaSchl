# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-3.0.0-py3-none-any.whl --force-reinstall

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature, set_signature
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.reservations_model import ReservationsModel

# COMMAND ----------
# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Define table names and function name
feature_table_name = f"{config.catalog_name}.{config.schema_name}.hotel_features"
function_name = f"{config.catalog_name}.{config.schema_name}.calculate_stay_length"


# COMMAND ----------
# Create or replace the hotel_features table
spark.sql(
    f"""
CREATE OR REPLACE TABLE {config.catalog_name}.{config.schema_name}.hotel_features
(Booking_ID STRING NOT NULL,
 no_of_previous_cancellations INT,
 avg_price_per_room FLOAT);
"""
)

spark.sql(
    f"ALTER TABLE {config.catalog_name}.{config.schema_name}.hotel_features "
    "ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);"
)

spark.sql(
    f"ALTER TABLE {config.catalog_name}.{config.schema_name}.hotel_features "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

# Insert data into the feature table from both train and test sets
spark.sql(
    f"INSERT INTO {config.catalog_name}.{config.schema_name}.hotel_features "
    f"SELECT Booking_ID, no_of_previous_cancellations, avg_price_per_room FROM {config.catalog_name}.{config.schema_name}.train_set_vs"
)
spark.sql(
    f"INSERT INTO {config.catalog_name}.{config.schema_name}.hotel_features "
    f"SELECT Booking_ID, no_of_previous_cancellations, avg_price_per_room FROM {config.catalog_name}.{config.schema_name}.test_set_vs"
)

# COMMAND ----------
# Define a function to calculate the overall stay length using the number of week and weekend nights
spark.sql(
    f"""
CREATE OR REPLACE FUNCTION {function_name}(no_of_week_nights INT, no_of_weekend_nights INT)
RETURNS INT
LANGUAGE PYTHON AS
$$
return no_of_week_nights + no_of_weekend_nights
$$
"""
)
# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set_vs").drop(
    "no_of_previous_cancellations", "avg_price_per_room"
)
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set_vs").toPandas()

# Cast no_of_weekend_nights and no_of_week_nights to int for the function input
train_set = train_set.withColumn("no_of_weekend_nights", train_set["no_of_weekend_nights"].cast("int"))
train_set = train_set.withColumn("no_of_week_nights", train_set["no_of_week_nights"].cast("int"))

# COMMAND ----------
# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["no_of_previous_cancellations", "avg_price_per_room"],
            lookup_key="Booking_ID",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="no_of_nights",
            input_bindings={
                "no_of_week_nights": "no_of_week_nights",
                "no_of_weekend_nights": "no_of_weekend_nights",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate no_of_nights for training and test set
test_set["no_of_nights"] = test_set["no_of_weekend_nights"] + test_set["no_of_week_nights"]

# COMMAND ----------
# Split features and target
y_train = training_df[[config.target]]
X_train = training_df.drop(columns=config.target)

y_test = test_set[[config.target]]
X_test = test_set.drop(columns=config.target)

# COMMAND ----------
# Setup model pipeline
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            DataProcessor(config=config, fe_features=["no_of_nights"]),
        ),
        ("classifier", ReservationsModel(config=config)),
    ]
)

# COMMAND ----------
# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-fe-vs")
GIT_SHA = "ffa63b430205ff7"

with mlflow.start_run(tags={"branch": "week1+2", "git_sha": f"{GIT_SHA}"}) as run:
    run_id = run.info.run_id
    y_train = y_train.replace({"Not_Canceled": "0", "Canceled": "1"}).astype(int).values
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_test = y_test.replace({"Not_Canceled": "0", "Canceled": "1"}).astype(int).values

    # Evaluate the model performance
    accuracy, precision = pipeline.named_steps["classifier"].evaluate(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "SVC with preprocessing")
    mlflow.log_params(config.parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        code_paths=["../hotel_reservations-3.0.0-py3-none-any.whl"],
        artifact_path="svc-pipeline-model-fe",
        training_set=training_set,
    )

    model_uri = f"runs:/{run_id}/svc-pipeline-model-fe"
    # set the signature for the logged model
    set_signature(model_uri, signature)


mlflow.register_model(
    model_uri=model_uri,
    name=f"{config.catalog_name}.{config.schema_name}.vs_hotel_reservations_model_fe",
)
