import numpy as np 
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class custom_logistic_regression :
    def __init__(self,X,Y,size): #constructor
        self.W = np.zeros(size) # initialize the weights (including the bias) as zeros
        self.X = X
        self.Y = Y
        self.optimal_threshold = 1 

    def Y_hat(self,i,X_to_predict: np.array): #returns Y hat (predicted value)
        safe_power = np.clip(-np.dot(X_to_predict[i], self.W), -250, 250)
        return (1 / (1 + np.exp(safe_power)))
    

    def derv_objective_func(self,xi:np.array,yi:np.array): #returns the derivative of the objective function
        safe_power = np.clip(yi * np.dot(xi, self.W), -250, 250)
        return ((-yi * xi) / (1 + np.exp(safe_power)))
    

    def estimate_optimal_threshold(self):
        pb = [] 
        for i in range(self.X.shape[0]):
            pb.append(self.Y_hat(i,self.X))
        
        FP_rates,TP_rates,Thresholds = roc_curve(self.Y,pb)
        Tp_Fp_rate_diff = TP_rates - FP_rates
        best_index = np.argmax(Tp_Fp_rate_diff)
        opt_Threshold = Thresholds[best_index]
        self.optimal_threshold = opt_Threshold
        print("Optimal threshold = ",self.optimal_threshold)

        #plotting-visualizing the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(FP_rates, TP_rates, color='blue', label='Custom Model ROC')
        plt.scatter(FP_rates[best_index], TP_rates[best_index], color='red', s=100, zorder=5,
                    label=f'Optimal Threshold ({opt_Threshold:.2f})')
        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
        plt.title('ROC Curve for my Custom Logistic Regression')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    #train the model using gradient descent
    def train_customized(self, _lambda_: float): 
        
        print("Training Started (Stochastic Gradient Descent)...")
        
        gdsc_iter = 100
        misclassification_list = []
        for iteration in range(gdsc_iter):
            misclassifications = 0 #track mistakes for early stopping
            
            for i in range(self.X.shape[0]):
                xi = self.X[i]
                yi = self.Y[i]
                
                err = self.derv_objective_func(xi, yi) 

                self.W -= _lambda_ * err #update weights every single entry error instead of total/batch error
                
                raw_score_instead_of_heavy_comp = np.dot(xi, self.W)
                predicted_class = 1 if raw_score_instead_of_heavy_comp >= 0 else -1
                
                if predicted_class != yi: #error encountered
                    misclassifications += 1
            misclassification_list.append(misclassifications)
                
            print(f"iteration {iteration} completed with errors: {misclassifications}")
            
            #early stop (reached convergence)
            if misclassifications == 0:
                print(f"Converged early at iteration: {iteration}")
                break
                
        print("training complete with weights: ", self.W)
        self.estimate_optimal_threshold()
        return (range(len(misclassification_list)),misclassification_list)


    def predict_customized(self,X_test): #classify the test data
        y_predicted = []
        for i in range(X_test.shape[0]):
            y_hat = self.Y_hat(i,X_test)
            if y_hat < self.optimal_threshold:
                y_predicted.append(-1)      
            else:
                y_predicted.append(1)
        
        return y_predicted