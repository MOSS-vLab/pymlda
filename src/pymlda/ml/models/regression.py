from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np

class SVRModel:
    
    def __init__(self):
        self.model = SVR(kernel='linear', C=10, epsilon=0.1)
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
    
    def fit(self, X, y):
        Xs = self.sc_X.fit_transform(X)
        ys = self.sc_y.fit_transform(y.reshape(-1,1))
        
        self.model.fit(Xs, ys.ravel())
        return self
    
    def predict(self, X):
        Xs = self.sc_X.transform(X)
        y_pred = self.model.predict(Xs)
        return self.sc_y.inverse_transform(y_pred.reshape(-1,1))