import numpy as np

class custom_naive_bayes :
    def __init__(self):

        self.Y_priors = {}
        self.Y_means = {}
        self.Y_variances = {}
        self.unique_labels = None
  

    def train(self,X_train,Y_train):
        self.unique_labels = np.unique(Y_train)

        for yi in self.unique_labels:
            #get data for this class only
            x_yi = X_train[Y_train == yi]
            self.Y_priors[yi]  = np.log(len(x_yi)/len(Y_train)) #calculate prior pb for each class
            
            self.Y_means[yi] = np.mean(x_yi, axis=0) #calculate mean for each feature in that class
            self.Y_variances[yi] = np.var(x_yi, axis=0) + 1e-9 #calculate variance for each feature in that class adding a small value to prevent division by zero later when predicting


    def predict(self,X_to_predict):
        y_hat = []
        for x in X_to_predict:
            yi_prob_like = {}
            for yi in self.unique_labels:
                mean = self.Y_means[yi]
                variance = self.Y_variances[yi]

                probabilities = -0.5*np.log(2 * np.pi * variance) - ((x-mean)**2/(2*variance))
                conditional_prob = np.sum(probabilities)
                #add log of the prior prob 
                yi_prob_like[yi] = conditional_prob + self.Y_priors[yi]
            
            # get the class with the highest probability
            higest_pr_class = max(yi_prob_like,key=yi_prob_like.get)
            y_hat.append(higest_pr_class)

        return np.array(y_hat)
            
