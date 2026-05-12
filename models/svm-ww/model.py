import numpy as np
import pandas as pd

class SVM_Multiclass:
    def __init__(self, x_train, y_train):
        raw_x = np.asarray(x_train)
        self.x_train = np.column_stack((raw_x, np.ones(raw_x.shape[0]))) 
        
        self.y_train = np.asarray(y_train, dtype=int).squeeze()
        
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
                sigmas[label] = -np.sum(sigmas)
                gradient_matrix = np.outer(x_i, sigmas)
                
                # --- WEIGHT UPDATE (Pegasos style) ---
                self.weights -= learning_rate * (gradient_matrix + (regularization * self.weights))
                
            print(f"Epoch {epoch+1}/{epochs} completed .......")

    def predict(self, x_test):
        x_test_np = np.asarray(x_test)
        x_test_aug = np.column_stack((x_test_np, np.ones(x_test_np.shape[0])))
        return np.argmax(x_test_aug @ self.weights, axis=1)