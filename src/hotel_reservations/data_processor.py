"""Data Processor class for the Hotel Reservations project."""

from copy import deepcopy

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_reservations.config import ProjectConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    """Class to preprocess and split data for the Hotel Reservations project."""

    def __init__(self, config: ProjectConfig, fe_features: list = None):
        if not isinstance(fe_features, list):
            fe_features = []
        self.config = config  # Store the configuration
        self.fe_features = fe_features  # Store the feature engineering features
        self.column_transformer = (
            ColumnTransformer(  # Initialize the column transformer
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        getattr(self.config, "cat_features"),
                    ),
                    ("scale", StandardScaler(), getattr(self.config, "num_features")),
                ],
                remainder="passthrough",
            )
        )

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | None = None
    ) -> BaseEstimator | TransformerMixin:
        """Fit method for the transformer."""
        self.column_transformer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with One-Hot Encoding and relevant feature extraction."""
        X = self.column_transformer.transform(X)
        # Extract relevant features
        X = self.extract_features(
            X=X, features="num_features", include_fe_features=True
        )
        return self.fix_column_names(X=X)

    def preprocess_data(
        self,
        X: pd.DataFrame,
        extract_features: str,
        include_fe_features: bool = True,
    ) -> pd.DataFrame:
        """Preprocess the DataFrame"""

        # Preprocess data with One-Hot Encoding and Scaling
        X = self.column_transformer.transform(X)
        X = self.fix_column_names(X=X)

        # Extract relevant features
        X = self.extract_features(
            X=X, features=extract_features, include_fe_features=include_fe_features
        )

        return X

    def fix_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the categorical features."""
        # One-hot-encode categorical features and fix column names
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
