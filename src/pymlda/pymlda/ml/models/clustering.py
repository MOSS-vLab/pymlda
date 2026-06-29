from sklearn.cluster import KMeans
import numpy as np

class KMeansClusterer:
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=0)
    
    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def elbow_method(self, X, max_k=10):
        wcss = []
        for i in range(1, max_k+1):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(X)
            wcss.append(km.inertia_)
        return wcss