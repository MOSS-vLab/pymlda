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

        # ADICIONAR AO FINAL DA CLASSE ClusteringFactory:

    @staticmethod
    def get_optimal_k(X, max_k=10, random_state=42):
        """
        Encontra o número ótimo de clusters usando o método do cotovelo.
        """
        wcss = ClusteringFactory.elbow_method(X, max_k, random_state)
        
        # Encontrar o ponto de máxima curvatura
        diffs = np.diff(wcss)
        diffs2 = np.diff(diffs)
        
        # Pular os primeiros pontos para evitar ruído
        start_idx = 1
        if len(diffs2) > start_idx:
            optimal_k = np.argmax(diffs2[start_idx:]) + start_idx + 2
        else:
            optimal_k = 2
        
        return max(2, min(optimal_k, max_k))
    
    @staticmethod
    def plot_clusters(X, labels, centroids=None, cluster_names=None, 
                      colors=None, title="Clustering Results", save_path=None):
        """
        Plota clusters com cores e nomes personalizados.
        """
        if colors is None:
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
                     'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
        
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        for i, cluster in enumerate(unique_labels):
            mask = labels == cluster
            color = colors[i % len(colors)]
            label = cluster_names[i] if cluster_names and i < len(cluster_names) else f'Cluster {cluster}'
            plt.scatter(X[mask, 0], X[mask, 1], s=100, c=color, 
                       edgecolor='white', linewidth=0.5, label=label)
        
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       s=300, c='red', marker='X', edgecolor='black', 
                       linewidth=2, label='Centróides')
        
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def get_cluster_info(X, labels):
        """Retorna informações sobre os clusters."""
        unique_labels = np.unique(labels)
        info = {}
        for cluster in unique_labels:
            mask = labels == cluster
            cluster_data = X[mask]
            info[cluster] = {
                'n_samples': len(cluster_data),
                'mean': cluster_data.mean(axis=0),
                'std': cluster_data.std(axis=0)
            }
        return info
    
    @staticmethod
    def print_cluster_info(X, labels):
        """Imprime informações sobre os clusters."""
        info = ClusteringFactory.get_cluster_info(X, labels)
        print("\n📊 Informações dos Clusters:")
        print("=" * 50)
        for cluster, data in info.items():
            print(f"\nCluster {cluster}:")
            print(f"  Nº de amostras: {data['n_samples']}")
            print(f"  Média: {data['mean']}")
            print(f"  Desvio: {data['std']}")