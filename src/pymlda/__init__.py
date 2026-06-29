# pymlda/__init__.py
"""
PyMLDA - Python Machine Learning for Damage Assessment
Pacote para monitoramento de saúde estrutural (SHM)
"""

__version__ = "0.2.0"

# Importações principais
from .ml.pipeline.ml_pipeline import MLDA
from .ml.models.classification import ClassifierFactory
from .ml.models.regression import RegressionFactory
from .ml.models.clustering import ClusteringFactory
from .ml.models.model_factory import get_models
from .ml.evaluation.metrics import compute_classification_metrics
from .ml.evaluation.validation import cross_validate
from .ml.evaluation.reports import generate_classification_report
from .ml.preprocessing.sampling import UnderSampler
from .ml.preprocessing.feature_selection import FeatureSelector
from .features.extractor import extract_features
from .features.windowed_extractor import extract_features_windowed
from .features.feature_manager import FeatureManager
from .utils.io import load_excel, save_dataframe
from .utils.data_split import split_data
from .utils.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_regression_results,
    plot_clusters,
    plot_learning_curves
)
from .utils.logging import get_logger

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
    'extract_features',
    'extract_features_windowed',
    'FeatureManager',
    'load_excel',
    'save_dataframe',
    'split_data',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_regression_results',
    'plot_clusters',
    'plot_learning_curves',
    'get_logger'
]