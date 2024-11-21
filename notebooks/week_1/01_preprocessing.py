# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-2.2.0-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import yaml

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.reservations_model import ReservationsModel
from hotel_reservations.utils import visualize_results

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataProcessor
df = spark.read.csv(
    "/Volumes/dev/datalab_1ai/files/hotel_reservations.csv",
    header=True,
    inferSchema=True,
).toPandas()
data_processor = DataProcessor(config=config)

# COMMAND ----------
# Split the data
train_set, test_set = data_processor.split_data(X=df)

print("Train set shape:", train_set.shape)
print("Test set shape:", test_set.shape)

# COMMAND ----------
# Split data into features and target
y_train = train_set[[config.original_target]]
X_train = train_set.drop(columns=config.original_target)

y_test = test_set[[config.original_target]]
X_test = test_set.drop(columns=config.original_target)

# COMMAND ----------
# Preprocess the data
X_train = data_processor.preprocess_data(
    X=X_train,
    encode_features="cat_features",
    extract_features="num_features",
    include_fe_features=True,
)
y_train = data_processor.preprocess_data(
    X=y_train,
    encode_features="original_target",
    extract_features="target",
    include_fe_features=False,
)
X_test = data_processor.preprocess_data(
    X=X_test,
    encode_features="cat_features",
    extract_features="num_features",
    include_fe_features=True,
)
y_test = data_processor.preprocess_data(
    X=y_test,
    encode_features="original_target",
    extract_features="target",
    include_fe_features=False,
)

# COMMAND ----------
# Initialize and train the model
model = ReservationsModel(config=config)
model.fit(X=X_train, y=y_train)

# COMMAND ----------
# Evaluate the model
y_pred = model.predict(X=X_test)
accuracy, precision = model.evaluate(y=y_test, y_pred=y_pred)
print(f"Accuracy: {round(accuracy, ndigits=4)}")
print(f"Precision: {round(precision, ndigits=4)}")

# COMMAND ----------
## Visualizing Results
y_pred = model.predict(X=X_test)
visualize_results(y_test=y_test, y_pred=y_pred)
