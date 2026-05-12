import numpy as np


class Metrics:

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

    """
        Builds a square matrix where rows are actual classes and columns are predicted classes.
        It will be a 10x10 grid for the 10 classes of teh mnist
    """

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Loop through every actual label and prediction pair
        for true_label, pred_label in zip(y_true, y_pred):
            # Increment the count in the corresponding grid cell
            matrix[int(true_label)][int(pred_label)] += 1

        return matrix

    @staticmethod
    def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):

        matrix = Metrics.confusion_matrix(y_true, y_pred, num_classes)

        precisions = []
        recalls = []
        f1_scores = []

        # Calculate metrics for each class (0 through 9) individually
        for c in range(num_classes):
            # True Positives: The diagonal element for this class
            TP = matrix[c, c]

            # False Positives: Everything else in this class's COLUMN
            FP = np.sum(matrix[:, c]) - TP

            # False Negatives: Everything else in this class's ROW
            FN = np.sum(matrix[c, :]) - TP

            # --- The Formulas (with your excellent division-by-zero safety checks!) ---
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Return the mean of all 10 classes
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)

        return avg_precision, avg_recall, avg_f1