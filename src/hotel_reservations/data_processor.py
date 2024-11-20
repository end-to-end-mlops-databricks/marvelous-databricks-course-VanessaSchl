"""Data Processor class for the Hotel Reservations project."""

from copy import deepcopy
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from hotel_reservations.config import ProjectConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    """Class to preprocess and split data for the Hotel Reservations project."""

    def __init__(
        self, config: ProjectConfig, fe_features: list = []
    ):
        self.config = config  # Store the configuration
        self.fe_features = fe_features  # Store the feature engineering features

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | None = None
    ) -> BaseEstimator | TransformerMixin:
        """Fit method for the transformer."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with One-Hot Encoding and relevant feature extraction."""
        X = self.one_hot_encode(X=X, features="cat_features")
        return self.extract_features(
            X=X, features="num_features", include_fe_features=True
        )

    def preprocess_data(
        self,
        X: pd.DataFrame,
        encode_features: str,
        extract_features: str,
        include_fe_features: bool = True,
    ) -> pd.DataFrame:
        """Preprocess the DataFrame"""

        # One-hot-encode categorical features and fix column names
        X = self.one_hot_encode(X=X, features=encode_features)

        # Extract target and relevant features
        X = self.extract_features(
            X=X, features=extract_features, include_fe_features=include_fe_features
        )

        return X

    def one_hot_encode(self, X: pd.DataFrame, features: str) -> pd.DataFrame:
        """One-hot encode the categorical features."""
        # One-hot-encode categorical features and fix column names
        cat_features = getattr(self.config, features)
        if not isinstance(cat_features, list):
            cat_features = [cat_features]
        X = pd.get_dummies(X, columns=cat_features)

        col_names = X.columns.to_list()
        for i, col in enumerate(col_names):
            col = col.replace(" ", "_").lower()
            col_names[i] = col
        X.columns = col_names

        return X

    def extract_features(
        self, X: pd.DataFrame, features: str, include_fe_features: bool = True
    ) -> pd.DataFrame:
        """Extract the target and relevant features."""
        num_features = getattr(self.config, features)

        relevant_columns = deepcopy(num_features)
        if include_fe_features:
            relevant_columns += self.fe_features

        return X[relevant_columns]

    def split_data(
        self, X: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        return train_set, test_set

    def save_to_catalog(self, spark: SparkSession, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set_vs"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set_vs"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set_vs "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set_vs "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
