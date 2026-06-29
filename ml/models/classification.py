# ml/models/classification.py
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ClassifierFactory:
    """Fábrica de classificadores para SHM"""
    
    @staticmethod
    def get(model_name="rf", **kwargs):
        """
        Retorna um classificador configurado.
        
        Parameters
        ----------
        model_name : str
            Nome do modelo: 'svm', 'knn', 'rf', 'dt', 'nb', 'xgb', 'gb'
        **kwargs : dict
            Parâmetros adicionais para o modelo
            
        Returns
        -------
        sklearn classifier
        """
        # Parâmetros padrão otimizados para SHM
        default_params = {
            'svm': {'kernel': 'rbf', 'C': 10, 'gamma': 'scale', 'probability': True, 'random_state': 42},
            'knn': {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'},
            'rf': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 
                   'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
            'dt': {'max_depth': 10, 'min_samples_split': 5, 'random_state': 42},
            'nb': {},  # GaussianNB não tem parâmetros importantes
            'xgb': {'objective': 'multi:softmax', 'n_estimators': 100, 
                    'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42},
            'gb': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 
                   'random_state': 42}
        }
        
        # Combina parâmetros padrão com os fornecidos
        params = default_params.get(model_name, {})
        params.update(kwargs)
        
        # Mapeamento de modelos
        models = {
            'svm': SVC(**params),
            'knn': KNeighborsClassifier(**params),
            'rf': RandomForestClassifier(**params),
            'dt': DecisionTreeClassifier(**params),
            'nb': GaussianNB(**params),
            'xgb': xgb.XGBClassifier(**params),
            'gb': GradientBoostingClassifier(**params)
        }
        
        if model_name not in models:
            raise ValueError(f"Modelo '{model_name}' não suportado. "
                           f"Opções: {list(models.keys())}")
        
        return models[model_name]
    
    @staticmethod
    def get_all():
        """Retorna todos os classificadores para ensemble"""
        models = {}
        for name in ['svm', 'knn', 'rf', 'dt', 'nb', 'xgb', 'gb']:
            try:
                models[name] = ClassifierFactory.get(name)
            except Exception as e:
                print(f"⚠️  Erro ao criar {name}: {e}")
        return models