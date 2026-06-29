from ml.models.classification_models import get_classification_models
from ml.models.regression_models import get_regression_model
from ml.models.clustering_models import get_clustering_model

def get_models():
    return {
        "classifiers": get_classification_models(),
        "regressor": get_regression_model(),
        "cluster": get_clustering_model()
    }