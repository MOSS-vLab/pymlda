# ml/models/model_factory.py
from .classification import ClassifierFactory
from .regression import RegressionFactory
from .clustering import ClusteringFactory

def get_models(task='classification', **kwargs):
    """
    Fábrica unificada para todos os modelos.
    
    Parameters
    ----------
    task : str
        'classification', 'regression', ou 'clustering'
    **kwargs : dict
        Parâmetros específicos do modelo
    
    Returns
    -------
    dict or sklearn model
    """
    if task == 'classification':
        return ClassifierFactory.get_all()
    elif task == 'regression':
        return RegressionFactory.get(**kwargs)
    elif task == 'clustering':
        return ClusteringFactory.get(**kwargs)
    else:
        raise ValueError(f"Task '{task}' não suportada. "
                        f"Opções: 'classification', 'regression', 'clustering'")

# Para compatibilidade com código existente
def get_classification_models():
    return ClassifierFactory.get_all()

def get_regression_model(model_name='rf', **kwargs):
    return RegressionFactory.get(model_name, **kwargs)

def get_clustering_model(model_name='kmeans', **kwargs):
    return ClusteringFactory.get(model_name, **kwargs)