import yaml
import logging

from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.reservations_model import ReservationsModel
from hotel_reservations.utils import visualize_results


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    spark = SparkSession.builder.getOrCreate()

    # Load configuration
    config = ProjectConfig.from_yaml(config_path='project_config.yml')

    logger.info("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # Load training and test sets
    train_set = spark.table(
        f"{config.catalog_name}.{config.schema_name}.train_set_vs"
    ).toPandas()
    test_set = spark.table(
        f"{config.catalog_name}.{config.schema_name}.test_set_vs"
    ).toPandas()

    # Split features and target
    y_train = train_set[[config.target]]
    X_train = train_set.drop(columns=[config.target])
    y_test = test_set[[config.target]]
    X_test = test_set.drop(columns=[config.target])
    logger.info(f"Data split into training and test sets.")
    logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Setup preprocessing and model pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", DataProcessor(config=config)),
            ("classifier", ReservationsModel(config=config)),
        ]
    )
    logger.info("Pipeline initialized.")

    # Train the model
    pipeline.fit(X=X_train, y=y_train)
    logger.info("Model training completed.")

    # Evaluate the model
    y_pred = pipeline.predict(X=X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    logger.info(f"Model evaluation completed: Accuracy={accuracy}, Precision={precision}, AUC={auc}")

    ## Visualizing Results
    visualize_results(y_test, y_pred)
    logger.info("Results visualization completed.")

if __name__ == "__main__":
  main()
