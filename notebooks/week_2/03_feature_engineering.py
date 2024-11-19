# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-1.1.4-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.reservations_model import ReservationsModel
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.pipeline import Pipeline

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

# Extract configuration details
num_features = config["num_features"]
target = config["target"]
parameters = config["parameters"]
catalog_name = config["catalog_name"]
schema_name = config["schema_name"]

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
function_name = f"{catalog_name}.{schema_name}.calculate_stay_length"


# COMMAND ----------
# Create or replace the hotel_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.hotel_features
(Booking_ID STRING NOT NULL,
 no_of_weekend_nights INT,
 avg_price_per_room FLOAT);
""")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.house_features "
          "ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.hotel_features "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data into the feature table from both train and test sets
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.hotel_features "
          f"SELECT no_of_weekend_nights, avg_price_per_room FROM {catalog_name}.{schema_name}.train_set_vs")
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.hotel_features "
          f"SELECT no_of_weekend_nights, avg_price_per_room FROM {catalog_name}.{schema_name}.test_set_vs")

# COMMAND ----------
# Define a function to calculate the overall stay length using the number of week and weekend nights
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(no_of_week_nights INT, no_of_weekend_nights INT)
RETURNS INT
LANGUAGE PYTHON AS
$$
return no_of_week_nights + no_of_weekend_nights
$$
""")
# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_vs").drop(
    "no_of_weekend_nights", "avg_price_per_room"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_vs").toPandas()

# Cast no_of_weekend_nights and no_of_week_nights to int for the function input
train_set = train_set.withColumn("no_of_weekend_nights", train_set["no_of_weekend_nights"].cast("int"))
train_set = train_set.withColumn("no_of_week_nights", train_set["no_of_week_nights"].cast("int"))

# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["no_of_weekend_nights", "avg_price_per_room"],
            lookup_key="Booking_ID",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="no_of_nights",
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate no_of_nights for training and test set
test_set["no_of_nights"] = test_set["no_of_weekend_nights"] + test_set["no_of_week_nights"]

# Split features and target
y_train = train_set[config["original_target"]]
X_train = train_set.drop(columns=config["original_target"])

y_test = test_set[config["original_target"]]
X_test = test_set.drop(columns=config["original_target"])

# Setup model pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", DataProcessor(config=config, spark=spark, fe_features=["no_of_nights"])),
        ("classifier", ReservationsModel(config=config))
    ]
)

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/house-prices-fe")
GIT_SHA = "ffa63b430205ff7"

with mlflow.start_run(tags={"branch": "week1+2",
                            "git_sha": f"{GIT_SHA}"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_test = pipeline.named_steps["preprocessor"].preprocess_data(
        X=y_test,
        encode_features="original_target",
        extract_features="target",
        include_fe_features=False
    )

    # Evaluate the model performance
    accuracy, precision = pipeline.named_steps["classifier"].evaluate(X_test, y_test)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "SVC with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="svc-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f'runs:/{run_id}/svc-pipeline-model-fe',
    name=f"{catalog_name}.{schema_name}.hotel_reservations_model_fe"
)