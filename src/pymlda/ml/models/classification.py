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
        """
        # Dicionário com os modelos
        model_classes = {
            'svm': SVC,
            'knn': KNeighborsClassifier,
            'rf': RandomForestClassifier,
            'dt': DecisionTreeClassifier,
            'nb': GaussianNB,
            'xgb': xgb.XGBClassifier,
            'gb': GradientBoostingClassifier
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Modelo '{model_name}' não suportado. "
                           f"Opções: {list(model_classes.keys())}")
        
        # Parâmetros padrão por modelo
        default_params = {
            'svm': {'kernel': 'rbf', 'C': 10, 'gamma': 'scale', 'probability': True, 'random_state': 42},
            'knn': {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'},
            'rf': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5,
                   'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
            'dt': {'max_depth': 10, 'min_samples_split': 5, 'random_state': 42},
            'nb': {},
            'xgb': {'objective': 'multi:softmax', 'n_estimators': 100,
                    'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42},
            'gb': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
                   'random_state': 42}
        }
        
        # Combinar parâmetros
        params = default_params.get(model_name, {}).copy()
        params.update(kwargs)
        
        # CORREÇÃO: Remove parâmetros inválidos para SVM
        if model_name == 'svm':
            invalid_params = ['n_estimators', 'max_depth', 'min_samples_split',
                            'n_jobs', 'class_weight', 'objective', 'learning_rate']
            for p in invalid_params:
                if p in params:
                    del params[p]
        
        # CORREÇÃO: Remove parâmetros inválidos para KNN
        if model_name == 'knn':
            invalid_params = ['n_estimators', 'max_depth', 'min_samples_split',
                            'random_state', 'n_jobs', 'class_weight', 'objective',
                            'learning_rate', 'gamma', 'C']
            for p in invalid_params:
                if p in params:
                    del params[p]
        
        # CORREÇÃO: Remove parâmetros inválidos para GaussianNB
        if model_name == 'nb':
            invalid_params = ['n_estimators', 'max_depth', 'min_samples_split',
                            'random_state', 'n_jobs', 'class_weight', 'objective',
                            'learning_rate', 'gamma', 'C', 'kernel', 'degree',
                            'coef0', 'probability', 'shrinking', 'tol', 'cache_size']
            for p in invalid_params:
                if p in params:
                    del params[p]
        
        # Criar e retornar o modelo
        return model_classes[model_name](**params)
    
    @staticmethod
    def get_all():
        """Retorna todos os classificadores"""
        models = {}
        for name in ['svm', 'knn', 'rf', 'dt', 'nb', 'xgb', 'gb']:
            try:
                models[name] = ClassifierFactory.get(name)
            except Exception as e:
                print(f"⚠️ Erro ao criar {name}: {e}")
        return models