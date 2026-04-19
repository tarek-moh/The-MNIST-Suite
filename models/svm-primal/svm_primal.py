import pandas as pd
import numpy as np

class SVMPrimal:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        raw_X = df.drop(columns='label').to_numpy()
        self.y = df['label'].to_numpy()
        
        self.X = np.column_stack((raw_X, np.ones(raw_X.shape[0])))
        
        self.w = np.zeros(self.X.shape[1])

    def train(self, learning_rate: float, epochs: int, reg: float):
        for epoch in range(epochs):
            
            errors = 0 
            print("Training Started...")
            for i in range(self.X.shape[0]): 
                point = self.X[i]
                label = self.y[i]

                if label * (self.w @ point) < 1:
                    self.w -= learning_rate * (reg * self.w - label * point)
                    errors += 1
                else:
                    self.w -= learning_rate * (reg * self.w)

            print(f"Epoch {epoch} completed ------- Errors: {errors}")
            if errors == 0: 
                print(f"Converged early at epoch {epoch}")
                break
        
    def predict(self, X_test: pd.DataFrame):
        raw_X_test = X_test.to_numpy()
        X_test_augmented = np.column_stack((raw_X_test, np.ones(raw_X_test.shape[0])))
        
        return np.sign(X_test_augmented @ self.w)