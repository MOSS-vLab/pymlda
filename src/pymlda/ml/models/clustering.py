# ml/models/clustering.py
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

class ClusteringFactory:
    """Fábrica de modelos de clustering para SHM"""
    
    @staticmethod
    def get(model_name="kmeans", **kwargs):
        """
        Retorna um clusterizador configurado.
        """
        # Dicionário com os modelos
        model_classes = {
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'agglomerative': AgglomerativeClustering,
            'gmm': GaussianMixture
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Modelo '{model_name}' não suportado. "
                           f"Opções: {list(model_classes.keys())}")
        
        # Parâmetros padrão por modelo
        default_params = {
            'kmeans': {'n_clusters': 4, 'random_state': 42, 'n_init': 10},
            'dbscan': {'eps': 0.5, 'min_samples': 5},  # <-- SEM 'n_clusters'
            'agglomerative': {'n_clusters': 4},
            'gmm': {'n_components': 4, 'random_state': 42}
        }
        
        # Combinar parâmetros
        params = default_params.get(model_name, {}).copy()
        params.update(kwargs)
        
        # CORREÇÃO: Remove parâmetros inválidos para DBSCAN
        if model_name == 'dbscan':
            invalid_params = ['n_clusters', 'n_components', 'random_state', 'n_init']
            for p in invalid_params:
                if p in params:
                    del params[p]
        
        # CORREÇÃO: Remove parâmetros inválidos para Agglomerative
        if model_name == 'agglomerative':
            if 'random_state' in params:
                del params['random_state']
        
        return model_classes[model_name](**params)
    
    @staticmethod
    def elbow_method(X, max_k=10, random_state=42):
        """Método do cotovelo para encontrar número ótimo de clusters"""
        wcss = []
        for i in range(1, max_k+1):
            km = KMeans(n_clusters=i, random_state=random_state, n_init=10)
            km.fit(X)
            wcss.append(km.inertia_)
        return wcss
    
    @staticmethod
    def plot_elbow(wcss, max_k=10):
        """Plota o gráfico do método do cotovelo"""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k+1), wcss, 'bo-')
        plt.xlabel('Número de Clusters')
        plt.ylabel('WCSS (Inércia)')
        plt.title('Método do Cotovelo')
        plt.grid(True, alpha=0.3)
        plt.show()