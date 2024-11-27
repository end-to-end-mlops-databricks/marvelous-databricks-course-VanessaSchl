"""Data Processor class for the Hotel Reservations project."""

import numpy as np
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
                        self.config.cat_features,
                    ),
                    ("scale_num", StandardScaler(), self.config.num_features),
                    ("scale_fe", StandardScaler(), fe_features),
                ],
                remainder="drop",
            )
        )

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | None = None
    ) -> BaseEstimator | TransformerMixin:
        """Fit method for the transformer."""
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess data with One-Hot Encoding and relevant feature extraction."""
        return self.column_transformer.transform(X)

    def split_data(
        self, X: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        return train_set, test_set
