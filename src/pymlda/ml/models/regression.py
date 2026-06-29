# ml/models/regression.py
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class RegressionFactory:
    """Fábrica de modelos de regressão para SHM"""
    
    @staticmethod
    def get(model_name="rf", **kwargs):
        """
        Retorna um regressor configurado.
        
        Parameters
        ----------
        model_name : str
            Nome do modelo: 'linear', 'ridge', 'lasso', 'elastic', 
            'svr', 'rf', 'gb'
        **kwargs : dict
            Parâmetros adicionais para o modelo
        """
        # Parâmetros padrão otimizados para SHM
        default_params = {
            'linear': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0, 'max_iter': 10000},
            'elastic': {'alpha': 1.0, 'l1_ratio': 0.5, 'max_iter': 10000},
            'svr': {'kernel': 'rbf', 'C': 10, 'epsilon': 0.1, 'gamma': 'scale'},
            'rf': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5,
                   'random_state': 42, 'n_jobs': -1},
            'gb': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
                   'random_state': 42}
        }
        
        # Combinar parâmetros padrão com os fornecidos
        params = default_params.get(model_name, {}).copy()
        params.update(kwargs)
        
        # Mapeamento de modelos
        models = {
            'linear': LinearRegression(**params),
            'ridge': Ridge(**params),
            'lasso': Lasso(**params),
            'elastic': ElasticNet(**params),
            'svr': SVR(**params),
            'rf': RandomForestRegressor(**params),
            'gb': GradientBoostingRegressor(**params)
        }
        
        if model_name not in models:
            raise ValueError(f"Modelo '{model_name}' não suportado. "
                           f"Opções: {list(models.keys())}")
        
        return models[model_name]
    
    @staticmethod
    def create_pipeline(model_name="rf", scaler=True, **kwargs):
        """Cria um pipeline com scaler opcional"""
        model = RegressionFactory.get(model_name, **kwargs)
        if scaler:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])
        return model

# Manter compatibilidade com versão anterior
# Criar uma instância padrão para SVRModel
SVRModel = RegressionFactory.get('svr')