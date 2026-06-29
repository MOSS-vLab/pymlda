# ml/pipeline/ml_pipeline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MLDA:
    """
    Pipeline completo para Structural Health Monitoring (SHM)
    
    Exemplo de uso:
    >>> from pymlda import MLDA
    >>> from pymlda.ml.models.classification import ClassifierFactory
    >>> from pymlda.ml.models.clustering import ClusteringFactory
    >>> 
    >>> # Configurar pipeline
    >>> pipeline = MLDA(
    ...     cluster_model=ClusteringFactory.get('kmeans', n_clusters=4),
    ...     classifiers=ClassifierFactory.get_all(),
    ...     regressor=RegressionFactory.get('rf'),
    ...     scaler=StandardScaler()
    ... )
    >>> 
    >>> # Treinar e avaliar
    >>> pipeline.fit(X_train, y_train)
    >>> results = pipeline.classify(X_test, y_test)
    >>> predictions = pipeline.regress(X_test)
    """
    
    def __init__(
        self,
        cluster_model=None,
        classifiers=None,
        regressor=None,
        scaler=None,
        random_state=42
    ):
        """
        Parameters
        ----------
        cluster_model : sklearn clusterer
            Modelo de clustering (ex: KMeans)
        classifiers : dict
            Dicionário de classificadores {nome: modelo}
        regressor : sklearn regressor
            Modelo de regressão
        scaler : sklearn scaler
            Scaler para normalização (padrão: StandardScaler)
        random_state : int
            Semente aleatória
        """
        self.cluster_model = cluster_model
        self.classifiers = classifiers or {}
        self.regressor = regressor
        self.scaler = scaler or StandardScaler()
        self.random_state = random_state
        
        # Dados
        self.X = None
        self.y = None
        self.X_scaled = None
        
        # Resultados
        self.cluster_labels = None
        self.X_clustered = None
        self.fitted_classifiers = {}
        self.trained_regressor = None
        
        # Métricas
        self.training_history = {
            'classification': {},
            'regression': {},
            'clustering': {}
        }
    
    def fit(self, X, y=None):
        """
        Armazena e prepara os dados.
        
        Parameters
        ----------
        X : array-like
            Matriz de features
        y : array-like, optional
            Labels (para classificação/regressão)
        """
        # Converter para DataFrame se necessário
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.X = X.copy()
        self.y = y
        
        # Normalizar
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Pipeline configurado com {X.shape[0]} amostras e {X.shape[1]} features")
        
        return self
    
    def cluster(self, X=None, method='fit_predict'):
        """
        Realiza clustering nos dados.
        
        Parameters
        ----------
        X : array-like, optional
            Dados para clusterizar. Se None, usa os dados de treino.
        method : str
            'fit_predict' (padrão) ou 'predict'
        
        Returns
        -------
        array : labels dos clusters
        """
        if self.cluster_model is None:
            raise RuntimeError("Nenhum modelo de clustering definido.")
        
        # Usar dados fornecidos ou os de treino
        if X is not None:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.X_scaled
        
        # Aplicar clustering
        if method == 'fit_predict':
            self.cluster_labels = self.cluster_model.fit_predict(X_scaled)
            if X is None:
                self.X_clustered = self.X.copy()
                self.X_clustered['cluster'] = self.cluster_labels
        else:
            self.cluster_labels = self.cluster_model.predict(X_scaled)
        
        # Estatísticas
        n_clusters = len(np.unique(self.cluster_labels))
        print(f"✅ Clustering concluído: {n_clusters} clusters identificados")
        
        return self.cluster_labels
    
    def classify(self, X=None, y=None):
        """
        Treina e avalia os classificadores.
        
        Parameters
        ----------
        X : array-like, optional
            Dados de teste. Se None, usa dados de treino.
        y : array-like, optional
            Labels de teste. Se None, usa labels de treino.
        
        Returns
        -------
        dict : Resultados da classificação
        """
        if not self.classifiers:
            raise RuntimeError("Nenhum classificador definido.")
        
        # Usar dados fornecidos ou os de treino
        if X is not None:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X_scaled = self.scaler.transform(X)
            y_true = y if y is not None else self.y
        else:
            X_scaled = self.X_scaled
            y_true = self.y
        
        if y_true is None:
            raise RuntimeError("Labels não fornecidos.")
        
        # Adicionar clusters se disponíveis
        if self.cluster_labels is not None:
            X_clustered = np.column_stack([X_scaled, self.cluster_labels])
        else:
            X_clustered = X_scaled
        
        results = {}
        self.fitted_classifiers = {}
        
        for name, model in self.classifiers.items():
            try:
                # Treinar
                model.fit(X_clustered, y_true)
                self.fitted_classifiers[name] = model
                
                # Prever
                y_pred = model.predict(X_clustered)
                
                # Métricas
                from ..evaluation.metrics import compute_classification_metrics
                metrics = compute_classification_metrics(y_true, y_pred)
                
                # Adicionar mais métricas
                from sklearn.metrics import accuracy_score, f1_score
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
                
                results[name] = metrics
                
                print(f"   ✅ {name}: Acurácia = {metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"   ❌ {name}: Erro - {e}")
                results[name] = {'error': str(e)}
        
        # Guardar histórico
        self.training_history['classification'] = results
        
        return results
    
    def regress(self, X=None, y=None):
        """
        Treina e aplica o modelo de regressão.
        
        Parameters
        ----------
        X : array-like, optional
            Dados para regressão. Se None, usa dados de treino.
        y : array-like, optional
            Target para treino. Se None, usa target de treino.
        
        Returns
        -------
        array : Previsões
        """
        if self.regressor is None:
            raise RuntimeError("Nenhum modelo de regressão definido.")
        
        # Usar dados fornecidos ou os de treino
        if X is not None:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X_scaled = self.scaler.transform(X)
            y_true = y if y is not None else self.y
        else:
            X_scaled = self.X_scaled
            y_true = self.y
        
        # Adicionar clusters se disponíveis
        if self.cluster_labels is not None:
            X_clustered = np.column_stack([X_scaled, self.cluster_labels])
        else:
            X_clustered = X_scaled
        
        # Treinar ou prever
        if y_true is not None:
            # Treinar
            self.regressor.fit(X_clustered, y_true)
            self.trained_regressor = self.regressor
            predictions = self.regressor.predict(X_clustered)
            
            # Métricas
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y_true, predictions),
                'r2': r2_score(y_true, predictions),
                'mae': mean_absolute_error(y_true, predictions)
            }
            self.training_history['regression'] = metrics
            print(f"✅ Regressão: R² = {metrics['r2']:.4f}, MSE = {metrics['mse']:.6f}")
        else:
            # Apenas prever
            if self.trained_regressor is None:
                raise RuntimeError("Modelo não treinado. Forneça y para treinar.")
            predictions = self.trained_regressor.predict(X_clustered)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Avaliação completa do pipeline.
        
        Parameters
        ----------
        X_test : array-like
            Dados de teste
        y_test : array-like
            Labels de teste
        
        Returns
        -------
        dict : Resultados completos
        """
        results = {
            'classification': {},
            'regression': {},
            'clustering': {}
        }
        
        # Classificação
        if self.classifiers:
            results['classification'] = self.classify(X_test, y_test)
        
        # Regressão
        if self.regressor is not None:
            pred = self.regress(X_test)
            if isinstance(y_test, (pd.Series, np.ndarray)):
                from sklearn.metrics import mean_squared_error, r2_score
                results['regression'] = {
                    'mse': mean_squared_error(y_test, pred),
                    'r2': r2_score(y_test, pred)
                }
        
        # Clustering
        if self.cluster_model is not None:
            results['clustering']['labels'] = self.cluster(X_test, method='predict')
        
        return results
    
    def cross_validate(self, X, y, cv=5, scoring='accuracy'):
        """
        Validação cruzada para classificação.
        """
        if not self.classifiers:
            raise RuntimeError("Nenhum classificador definido.")
        
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        for name, model in self.classifiers.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
                results[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                print(f"   ✅ {name}: {scores.mean():.4f} (±{scores.std():.4f})")
            except Exception as e:
                print(f"   ❌ {name}: Erro - {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def summary(self):
        """Resumo do pipeline"""
        print("\n" + "="*60)
        print("📊 RESUMO DO PIPELINE")
        print("="*60)
        
        if self.X is not None:
            print(f"Dados: {self.X.shape[0]} amostras, {self.X.shape[1]} features")
        
        if self.cluster_model is not None:
            print(f"Clustering: {type(self.cluster_model).__name__}")
            if self.cluster_labels is not None:
                n_clusters = len(np.unique(self.cluster_labels))
                print(f"  - Clusters: {n_clusters}")
        
        if self.classifiers:
            print(f"Classificadores: {len(self.classifiers)}")
            for name in self.classifiers.keys():
                print(f"  - {name}")
            if self.training_history['classification']:
                print("  - Resultados:")
                for name, metrics in self.training_history['classification'].items():
                    if 'accuracy' in metrics:
                        print(f"    * {name}: {metrics['accuracy']:.4f}")
        
        if self.regressor is not None:
            print(f"Regressão: {type(self.regressor).__name__}")
            if self.training_history['regression']:
                metrics = self.training_history['regression']
                print(f"  - R²: {metrics.get('r2', 'N/A'):.4f}")
                print(f"  - MSE: {metrics.get('mse', 'N/A'):.6f}")
        
        print("="*60)

    # Adicionar ao final da classe MLDA (dentro do arquivo ml_pipeline.py)

    def from_features(self, X, y=None):
        """
        Configura o pipeline usando features já extraídas.
        Útil quando você já tem as features e não precisa reextraí-las.
        
        Parameters
        ----------
        X : array-like
            Matriz de features já extraídas
        y : array-like, optional
            Labels (para classificação/regressão)
        
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.X = X.copy()
        self.y = y
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Pipeline configurado com {X.shape[0]} amostras e {X.shape[1]} features (pré-calculadas)")
        
        return self