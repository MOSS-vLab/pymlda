# ml/preprocessing/feature_selection.py
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE, RFECV
import pandas as pd
import numpy as np

class FeatureSelector:
    """Seleção de features para SHM"""
    
    @staticmethod
    def select_k_best(X, y, k=5, score_func=f_classif):
        """Seleciona as k melhores features"""
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector
    
    @staticmethod
    def mutual_info(X, y, k=5):
        """Seleção por informação mútua"""
        return FeatureSelector.select_k_best(X, y, k, mutual_info_classif)
    
    @staticmethod
    def rfe(model, X, y, n_features=5):
        """Recursive Feature Elimination"""
        selector = RFE(model, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector
    
    @staticmethod
    def get_feature_importance(model, feature_names=None):
        """Obtém importância das features"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            raise ValueError("Modelo não suporta importância de features")
        
        if feature_names is not None:
            return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        return importances