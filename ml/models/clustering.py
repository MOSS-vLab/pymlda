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
        
        Parameters
        ----------
        model_name : str
            Nome do modelo: 'kmeans', 'dbscan', 'agglomerative', 'gmm'
        **kwargs : dict
            Parâmetros adicionais para o modelo
        """
        default_params = {
            'kmeans': {'n_clusters': 4, 'random_state': 42, 'n_init': 10},
            'dbscan': {'eps': 0.5, 'min_samples': 5},
            'agglomerative': {'n_clusters': 4, 'linkage': 'ward'},
            'gmm': {'n_components': 4, 'random_state': 42}
        }
        
        params = default_params.get(model_name, {})
        params.update(kwargs)
        
        models = {
            'kmeans': KMeans(**params),
            'dbscan': DBSCAN(**params),
            'agglomerative': AgglomerativeClustering(**params),
            'gmm': GaussianMixture(**params)
        }
        
        if model_name not in models:
            raise ValueError(f"Modelo '{model_name}' não suportado. "
                           f"Opções: {list(models.keys())}")
        
        return models[model_name]
    
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
        plt.title('Método do Cotovelo para Clustering')
        plt.grid(True, alpha=0.3)
        plt.show()

# Manter compatibilidade
KMeansClusterer = ClusteringFactory.get('kmeans')