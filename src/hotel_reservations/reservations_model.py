"""Model for predicting hotel cancellations."""

from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler


class ReservationsModel(BaseEstimator, ClassifierMixin):
    """Class to train and evaluate a model for predicting cancellation on hotel reservations."""
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.model = svm.SVC(
                C=config['parameters']['C'],
                kernel=config['parameters']['kernel'],
                gamma=config['parameters']['gamma'],
                tol=config['parameters']['tol'],
                max_iter=config['parameters']['max_iter'],
                random_state=42
        )

    def fit(self, X, y):
        """Fit the model to the training data."""
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the trained model."""
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model on the test data."""
        X = self.scaler.transform(X)
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        return accuracy, precision
