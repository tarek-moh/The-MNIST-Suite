from sklearn.datasets import fetch_openml
import numpy as np
from collections import Counter

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """Memorizes the training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """Predicts the class for eac   h input instance."""
        X = np.array(X)
        predictions = []
        for x in X:
            # Calculate Euclidean distance using NumPy broadcasting
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Get indices of the top k closest training samples
            k_indices = np.argsort(distances)[:self.k]

            # Retrieve their corresponding labels
            k_nearest_labels = self.y_train[k_indices]

            # Determine the majority class
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)

def KNN_init():
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)

    model = CustomKNN(k=5)
    model.fit(X, y)
    return model