"""Model for predicting hotel cancellations."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score
from xgboost.sklearn import XGBClassifier

from hotel_reservations.config import ProjectConfig


class ReservationsModel(BaseEstimator, ClassifierMixin):
    """Class to train and evaluate a model for predicting cancellation on hotel reservations."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.model = XGBClassifier(
            eta=self.config.parameters["eta"],
            n_estimators=self.config.parameters["n_estimators"],
            max_depth=self.config.parameters["max_depth"],
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the training data."""
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray):
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def evaluate(self, y: np.ndarray, y_pred: np.ndarray):
        """Evaluate the model on the test data."""
        accuracy = accuracy_score(y_true=y, y_pred=y_pred)
        precision = precision_score(y_true=y, y_pred=y_pred)
        return accuracy, precision
