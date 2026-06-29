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
        """
        # Dicionário com os modelos (sem parâmetros)
        models = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elastic': ElasticNet,
            'svr': SVR,
            'rf': RandomForestRegressor,
            'gb': GradientBoostingRegressor
        }
        
        if model_name not in models:
            raise ValueError(f"Modelo '{model_name}' não suportado. "
                           f"Opções: {list(models.keys())}")
        
        # Parâmetros padrão por modelo
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
        
        # Combinar parâmetros
        params = default_params.get(model_name, {}).copy()
        params.update(kwargs)
        
        # ============================================================
        # CORREÇÃO: Remove parâmetros inválidos para cada modelo
        # ============================================================
        
        # Para LinearRegression - remove TODOS os parâmetros que não são aceitos
        if model_name == 'linear':
            # Lista de parâmetros que o LinearRegression NÃO aceita
            invalid_params = [
                'kernel', 'gamma', 'C', 'epsilon', 'n_estimators', 
                'max_depth', 'min_samples_split', 'random_state', 'n_jobs',
                'learning_rate', 'alpha', 'l1_ratio', 'max_iter',
                'degree', 'coef0', 'tol', 'cache_size', 'verbose',
                'class_weight', 'decision_function_shape', 'break_ties',
                'shrinking', 'probability', 'warm_start'
            ]
            for p in invalid_params:
                if p in params:
                    del params[p]
                    print(f"   🔄 Removendo parâmetro inválido: {p}")
        
        # Para Ridge, Lasso, ElasticNet - remove parâmetros de outros modelos
        if model_name in ['ridge', 'lasso', 'elastic']:
            invalid_params = ['kernel', 'gamma', 'C', 'epsilon', 'n_estimators',
                            'max_depth', 'min_samples_split', 'random_state', 'n_jobs',
                            'learning_rate', 'degree', 'coef0', 'tol']
            for p in invalid_params:
                if p in params:
                    del params[p]
        
        # Para SVR - remove parâmetros de árvores
        if model_name == 'svr':
            invalid_params = ['n_estimators', 'max_depth', 'min_samples_split',
                            'random_state', 'n_jobs', 'learning_rate', 'alpha',
                            'l1_ratio', 'max_iter']
            for p in invalid_params:
                if p in params:
                    del params[p]
        
        # Criar e retornar o modelo
        return models[model_name](**params)
    
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
SVRModel = RegressionFactory.get('svr')