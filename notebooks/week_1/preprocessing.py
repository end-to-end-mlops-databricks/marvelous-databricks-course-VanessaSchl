# Databricks notebook source
# MAGIC %pip install ../hotel_reservations-0.0.2-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import pandas as pd
import yaml
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.reservations_model import ReservationsModel
from hotel_reservations.utils import plot_feature_importance, visualize_results

# Load configuration
with open("../../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataProcessor
pandas_df = pd.read_csv("/Volumes/dev/datalab_1ai/files/hotel_reservations.csv")
data_processor = DataProcessor(pandas_df, config)

# Preprocess the data
data_processor.preprocess_data()

# COMMAND ----------
# Split the data
train_set, test_set = data_processor.split_data()

print("Train set shape:", train_set.shape)
print("Test set shape:", test_set.shape)

# COMMAND ----------
# Split data into features and target
X_train = train_set[config["num_features"]]
y_train = train_set[config["target"]]

X_test = test_set[config["num_features"]]
y_test = test_set[config["target"]]

# COMMAND ----------
# Initialize and train the model
model = ReservationsModel(data_processor.preprocess_data, config)
model.train(X_train, y_train)

# COMMAND ----------
# Evaluate the model
accuracy, precision = model.evaluate(X_test, y_test)
print(f"Accuracy: {round(accuracy, ndigits=4)}")
print(f"Precision: {round(precision, ndigits=4)}")

# COMMAND ----------
## Visualizing Results
y_pred = model.predict(X_test)
visualize_results(y_test, y_pred)

# COMMAND ----------
## Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)
