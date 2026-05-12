import numpy as np
import pandas as pd
from collections import Counter

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from common.preprocessing_util import PreprocessingUtil as pp
from sklearn.model_selection import train_test_split, KFold

# ==========================================
# 1. CUSTOM KNN IMPLEMENTATION
# ==========================================
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


# ==========================================
# 2. CROSS-VALIDATION FOR PARAMETER TUNING
# ==========================================

def cross_validate_knn(X, y, k_values, n_splits=5, scoring='accuracy'):
    """
    Performs K-Fold cross-validation over a list of k values.

    Args:
        X         : Feature matrix (numpy array).
        y         : Labels (numpy array).
        k_values  : List of k values to evaluate.
        n_splits  : Number of folds (default: 5).
        scoring   : Metric to optimize — 'accuracy', 'f1', 'precision', or 'recall'.

    Returns:
        cv_results : Dict mapping each k to its per-fold scores and mean/std.
        best_k     : The k value with the highest mean CV score.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = {}

    scoring_fn = {
        'accuracy':  lambda yt, yp: Metrics.accuracy(yt, yp), # accuracy is already an average
        'f1':        lambda yt, yp: Metrics.evaluate_all(yt, yp)[2],
        'precision': lambda yt, yp: Metrics.evaluate_all(yt, yp)[0],
        'recall':    lambda yt, yp: Metrics.evaluate_all(yt, yp)[1],
    }[scoring]

    print(f"\n--- Cross-Validation ({n_splits}-Fold, metric='{scoring}') ---")
    print(f"{'k':<6} {'Fold Scores':<50} {'Mean':>8} {'Std':>8}")
    print("-" * 75)

    for k in k_values:
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            fold_scaler = StandardScaler()
            X_fold_train = fold_scaler.fit_transform(X_fold_train)
            X_fold_val   = fold_scaler.transform(X_fold_val)

            model = CustomKNN(k=k)
            model.fit(X_fold_train, y_fold_train)
            y_val_pred = model.predict(X_fold_val)

            score = scoring_fn(y_fold_val, y_val_pred)
            fold_scores.append(score)
            print(f"for k = {k} score = {score}")

        mean_score = np.mean(fold_scores)
        std_score  = np.std(fold_scores)
        cv_results[k] = {'fold_scores': fold_scores, 'mean': mean_score, 'std': std_score}

        scores_str = '  '.join([f'{s:.4f}' for s in fold_scores])
        print(f"k={k:<4} [{scores_str}]  {mean_score:.4f}   ±{std_score:.4f}")

    best_k = max(cv_results, key=lambda k: cv_results[k]['mean'])
    print(f"\n✓ Best k={best_k}  (mean {scoring}={cv_results[best_k]['mean']:.4f})")
    return cv_results, best_k


# ==========================================
# 3. DATA PROCESSING & PIPELINE
# ==========================================

print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)

# Train / Test Split — scaler is NOT applied here; CV handles it fold-by-fold
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Feature Extraction Pipeline ---
FEATURE_STRATEGY = 'hog'

def extract_features(data, strategy='flatten'):
    if strategy == 'flatten':
        return data
    elif strategy == 'hog':
        return pp.hog_feature_extractor(pd.DataFrame(data))

print(f"Extracting features using {FEATURE_STRATEGY.upper()}...")
X_train_feat = extract_features(X_train, strategy=FEATURE_STRATEGY)
X_test_feat  = extract_features(X_test,  strategy=FEATURE_STRATEGY)

X_train_feat = np.array(X_train_feat)
X_test_feat  = np.array(X_test_feat)

# ==========================================
# 4. CROSS-VALIDATION: FIND BEST K
# ==========================================

# Use 20% of training data for CV — enough to reliably rank k values
X_cv, _, y_cv, _ = train_test_split(
    X_train_feat, y_train,
    train_size=0.2,
    random_state=42,
    stratify=y_train
)

K_CANDIDATES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
CV_METRIC     = 'f1' # 'accuracy', 'f1', 'precision', 'recall'
N_FOLDS       = 5

cv_results, best_k = cross_validate_knn(
    X_cv, y_cv,
    k_values=K_CANDIDATES,
    n_splits=N_FOLDS,
    scoring=CV_METRIC,
)


# ==========================================
# 5. FINAL TRAINING WITH BEST K
# ==========================================

# Scale using the full training set now that k is fixed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled  = scaler.transform(X_test_feat)

print(f"\nTraining final model with best k={best_k} on full training set...")
final_model = CustomKNN(k=best_k)
final_model.fit(X_train_scaled, y_train)

print("Running predictions on the test set...")
y_pred = final_model.predict(X_test_scaled)

# ==========================================
# 6. EVALUATION
# ==========================================

from common.metrices import Metrics

print("\n--- Final Model Evaluation ---")
accuracy = Metrics.accuracy(y_test, y_pred)
precision, recall, F1 = Metrics.evaluate_all(y_test, y_pred)
conf_matrix = Metrics.confusion_matrix(y_test, y_pred)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:  {recall:.4f}")
print(f"F1-Score:  {F1:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)


# ==========================================
# 7. Load CNN data
# ==========================================

X_cnn = np.load("../../common/CNN_data/X.npy")
y_cnn = np.load("../../common/CNN_data/Y.npy")

# Train / Test Split
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 8. Train the model
# ==========================================

print(f"\nTraining model with k={best_k} on CNN dataset...")
final_model = CustomKNN(k=best_k)
final_model.fit(X_train_cnn, y_train_cnn)

# ==========================================
# 9. Test the model
# ==========================================

print("Running predictions on the test set...")
y_pred_cnn = final_model.predict(X_test_cnn)

# ==========================================
# 10. EVALUATION
# ==========================================

from common.metrices import Metrics

print("\n--- Final Model Evaluation ---")
accuracy = Metrics.accuracy(y_test_cnn, y_pred_cnn)
precision, recall, F1 = Metrics.evaluate_all(y_test_cnn, y_pred_cnn)
conf_matrix = Metrics.confusion_matrix(y_test_cnn, y_pred_cnn)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:  {recall:.4f}")
print(f"F1-Score:  {F1:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)