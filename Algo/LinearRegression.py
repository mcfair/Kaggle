from __future__ import print_function
import numpy as np

class LinearRegression(object):
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __loss(self, h, y):
        """ MSE loss """
        return np.mean(np.power(y-h, 2))
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        # weight initialization
        self.weight = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            h = np.dot(X, self.weight)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weight -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                h = np.dot(X, self.weight)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.weight))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

 if __name__ == '__main__':
    model = LinearRegression(lr=0.1, num_iter=300000)
    pred_y = model.predict(X)
    print(np.mean(pred_y == y)) # accuracy
