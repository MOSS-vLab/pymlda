from sklearn.metrics import classification_report

def generate_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)

# ADICIONAR AO FINAL DO ARQUIVO reports.py:

import pandas as pd

def generate_comparison_table(results, metrics=['accuracy', 'precision', 'recall', 'f1']):
    """
    Gera uma tabela comparativa de modelos.
    """
    table = {}
    for model_name, model_results in results.items():
        row = {}
        for metric in metrics:
            if metric in model_results:
                row[metric] = model_results[metric]
        table[model_name] = row
    return pd.DataFrame(table).T

def save_results_to_excel(results, filename='results.xlsx'):
    """
    Salva resultados em Excel.
    """
    with pd.ExcelWriter(filename) as writer:
        for model_name, metrics in results.items():
            if isinstance(metrics, dict):
                df = pd.DataFrame([metrics])
                sheet_name = model_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    return filename