"""Utility functions for the hotel reservations project."""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_results(y_test, y_pred):
    """Visualize the results of the model predictions."""
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize="all")

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.2f',
        vmin=0.0,
        vmax=1.0,
        xticklabels=['Not Cancelled', 'Cancelled'],
        yticklabels=['Not Cancelled', 'Cancelled']
    )

    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.show()
