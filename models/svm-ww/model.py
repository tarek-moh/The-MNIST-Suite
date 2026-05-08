import numpy as np
import pandas as pd

class SVM_Multiclass:
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self.x_train = x_train.to_numpy()
        self.x_train = np.column_stack((self.x_train, np.ones(self.x_train.shape[0]))) 
        self.y_train = y_train.to_numpy()
        
        self.num_classes = len(np.unique(self.y_train)) 
        
    def train(self, epochs, learning_rate=0.01, regularization=0.01):
        self.weights = np.zeros((self.x_train.shape[1], self.num_classes)) 

        for epoch in range(epochs):
            for j in range(self.x_train.shape[0]):
                x_i = self.x_train[j]
                label = self.y_train[j]
                
                # --- FORWARD PASS ---
                WTX = x_i @ self.weights 
                true_class_score = WTX[label] 
                margins = np.maximum(0, 1 + WTX - true_class_score)
                margins[label] = 0 
                sigmas = (margins > 0).astype(int)
                
                # --- GRADIENT LOGIC ---
                # The true class gradient is the negative sum of the rival violations
                sigmas[label] = -np.sum(sigmas)
                
                # I want to scale X vector by each sigma, but I want gradients of same class to be columns
                # so I transpose
                gradient_matrix = np.outer(x_i, sigmas)
                
                # --- WEIGHT UPDATE (Pegasos style) ---
                # Subtract learning_rate * (Gradient + L2 Regularization Penalty)
                self.weights -= learning_rate * (gradient_matrix + (regularization * self.weights))


    def predict(self, x_test):
        x_test_np = x_test.to_numpy()
        x_test_aug = np.column_stack((x_test_np, np.ones(x_test_np.shape[0])))
        return np.argmax(x_test_aug @ self.weights, axis=1)
