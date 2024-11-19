"""Model for predicting hotel cancellations."""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler

from hotel_reservations.config import ProjectConfig


class ReservationsModel(BaseEstimator, ClassifierMixin):
    """Class to train and evaluate a model for predicting cancellation on hotel reservations."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            eta=self.config.parameters["eta"],
            n_estimators=self.config.parameters["n_estimators"],
            max_depth=self.config.parameters["max_depth"],
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the model to the training data."""
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        """Make predictions using the trained model."""
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame):
        """Evaluate the model on the test data."""
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        return accuracy, precision