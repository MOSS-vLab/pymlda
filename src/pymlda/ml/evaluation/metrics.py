from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)

def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="micro"),
        "precision": precision_score(y_true, y_pred, average="micro"),
        "recall": recall_score(y_true, y_pred, average="micro")
    }
# ADICIONAR AO FINAL DO ARQUIVO metrics.py:

def get_feature_importance(model, feature_names=None, top_n=10, plot=True):
    """
    Extrai e plota a importância das features.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Modelo não suporta importância de features")
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    indices = np.argsort(importances)[::-1][:top_n]
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], 
                   rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importância')
        plt.title(f'Top {top_n} Features Mais Importantes')
        plt.tight_layout()
        plt.show()
    
    return {feature_names[i]: importances[i] for i in indices}

def regression_by_group(y_true, y_pred, groups, group_names=None):
    """
    Analisa erros de regressão agrupados por classe.
    """
    import pandas as pd
    
    results = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': groups
    })
    
    metrics = {}
    for group in results['group'].unique():
        mask = results['group'] == group
        y_true_g = results.loc[mask, 'y_true']
        y_pred_g = results.loc[mask, 'y_pred']
        
        from sklearn.metrics import mean_squared_error, r2_score
        metrics[group] = {
            'n_samples': len(y_true_g),
            'mean_true': y_true_g.mean(),
            'mean_pred': y_pred_g.mean(),
            'mse': mean_squared_error(y_true_g, y_pred_g),
            'r2': r2_score(y_true_g, y_pred_g)
        }
    
    return metrics, results