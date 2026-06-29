# ml/__init__.py
"""
Módulo de Machine Learning para SHM
"""

from .pipeline.ml_pipeline import MLDA
from .models.classification import ClassifierFactory
from .models.regression import RegressionFactory
from .models.clustering import ClusteringFactory
from .models.model_factory import get_models
from .evaluation.metrics import compute_classification_metrics
from .evaluation.validation import cross_validate
from .evaluation.reports import generate_classification_report
from .preprocessing.sampling import UnderSampler
from .preprocessing.feature_selection import FeatureSelector
from .preprocessing.time_series import TimeSeriesProcessor

__all__ = [
    'MLDA',
    'ClassifierFactory',
    'RegressionFactory',
    'ClusteringFactory',
    'get_models',
    'compute_classification_metrics',
    'cross_validate',
    'generate_classification_report',
    'UnderSampler',
    'FeatureSelector',
    'TimeSeriesProcessor'
]