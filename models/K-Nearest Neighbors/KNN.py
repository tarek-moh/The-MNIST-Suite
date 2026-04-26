import numpy as np
from collections import Counter
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skimage.feature import hog

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
        """Predicts the class for each input instance."""
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
# 2. DATA PROCESSING & PIPELINE
# ==========================================

print("Loading MNIST dataset...")
# Load MNIST (This may take a minute)
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)

# Binary Classification Filter
binary_mask = (y == 1) | (y == 4) | (y == 7)
y_binary = np.ones_like(y)
y_binary[binary_mask] = 0

print(f"Total samples in class 0: {X[binary_mask].shape[0]}")
print(f"Total samples in class 1: {X.shape[0] - X[binary_mask].shape[0]}")

# --- Train / Validation / Test Split ---
# First split: 80% Train, 20% Temp (to be split into Val/Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_binary, test_size=0.2, random_state=42,
                                                    stratify=y_binary)

# Second split: Divide Temp evenly to get 10% Validation, 10% Test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# --- Feature Extraction Pipeline ---
# You can choose between 'flatten' (raw pixels), 'pca', or 'hog'
FEATURE_STRATEGY = 'pca'  # Try changing this to 'flatten' or 'hog'


def extract_features(data, strategy='flatten', pca_model=None):
    if strategy == 'flatten':
        # MNIST is already flattened in the scikit-learn dataset (784 features)
        return data
    elif strategy == 'pca':
        if pca_model is None:
            # Fit PCA to keep 95% of variance
            pca_model = PCA(n_components=0.95)
            data_transformed = pca_model.fit_transform(data)
            return data_transformed, pca_model
        return pca_model.transform(data)
    elif strategy == 'hog':
        # Reshape to 28x28 for HOG extraction
        features = []
        for img in data:
            image_2d = img.reshape((28, 28))
            fd = hog(image_2d, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=False)
            features.append(fd)
        return np.array(features)


print(f"Extracting features using {FEATURE_STRATEGY.upper()}...")
if FEATURE_STRATEGY == 'pca':
    X_train_feat, pca_model = extract_features(X_train, strategy=FEATURE_STRATEGY)
    X_val_feat = extract_features(X_val, strategy=FEATURE_STRATEGY, pca_model=pca_model)
    X_test_feat = extract_features(X_test, strategy=FEATURE_STRATEGY, pca_model=pca_model)
else:
    X_train_feat = extract_features(X_train, strategy=FEATURE_STRATEGY)
    X_val_feat = extract_features(X_val, strategy=FEATURE_STRATEGY)
    X_test_feat = extract_features(X_test, strategy=FEATURE_STRATEGY)

# --- Feature Normalization ---
# Critical for distance-based algorithms like KNN to ensure all features weigh equally.
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_val_feat = scaler.transform(X_val_feat)
X_test_feat = scaler.transform(X_test_feat)

# ==========================================
# 3. TRAINING AND EVALUATION
# ==========================================

# Initialize our custom model
k_value = 5
print(f"Training Custom KNN with k={k_value}...")
model = CustomKNN(k=k_value)

# Fit (Store the data)
model.fit(X_train_feat, y_train)

# Predict on the Test Set
# (In a real scenario, you'd use the Validation set here to tune 'k' first)
print("Running predictions on the test set...")
y_pred = model.predict(X_test_feat)

# --- Evaluation Metrics ---
print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))