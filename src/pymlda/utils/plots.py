# utils/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_confusion_matrix(cm, labels=None, title="Matriz de Confusão", figsize=(8, 6)):
    """Plota matriz de confusão melhorada"""
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importances, feature_names, title="Importância das Features", top_n=10):
    """Plota importância das features"""
    if isinstance(importances, dict):
        importances = pd.Series(importances)
    
    # Ordenar
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_idx)), importances[sorted_idx])
    plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importância')
    plt.tight_layout()
    plt.show()

def plot_regression_results(y_true, y_pred, title="Resultados da Regressão"):
    """Plota resultados de regressão"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de dispersão
    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Valores Reais')
    axes[0].set_ylabel('Valores Previstos')
    axes[0].set_title('Previsto vs Real')
    axes[0].grid(True, alpha=0.3)
    
    # Resíduos
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Valores Previstos')
    axes[1].set_ylabel('Resíduos')
    axes[1].set_title('Análise de Resíduos')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Estatísticas dos resíduos
    print(f"Média dos resíduos: {np.mean(residuals):.6f}")
    print(f"Desvio padrão dos resíduos: {np.std(residuals):.6f}")

def plot_clusters(X, labels, centers=None, title="Resultado do Clustering"):
    """Plota resultados de clustering"""
    plt.figure(figsize=(10, 7))
    
    # Plotar pontos
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    
    # Plotar centróides
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, 
                   edgecolors='white', linewidth=2, label='Centróides')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history, title="Curvas de Aprendizado"):
    """Plota curvas de aprendizado para redes neurais"""
    plt.figure(figsize=(12, 4))
    
    if 'train_loss' in history and 'val_loss' in history:
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Treino')
        plt.plot(history['val_loss'], label='Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.title('Curva de Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    if 'train_acc' in history and 'val_acc' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Treino')
        plt.plot(history['val_acc'], label='Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.title('Curva de Acurácia')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# DENSITY PLOT
# ============================================================

def plot_density_by_group(data, group_column, value_column, 
                          group_names=None, colors=None,
                          title="Density Plot", save_path=None, figsize=(20, 5)):
    """
    Plota densidades de uma variável agrupada por classe.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame com os dados
    group_column : str
        Nome da coluna com os grupos
    value_column : str
        Nome da coluna com os valores
    group_names : list, optional
        Nomes personalizados para os grupos
    colors : list, optional
        Cores para os grupos
    title : str
        Título do gráfico
    save_path : str, optional
        Caminho para salvar a figura
    figsize : tuple
        Tamanho da figura (padrão: (20, 5))
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    groups = data[group_column].unique()
    n_groups = len(groups)
    
    if colors is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
                  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    
    # Ajustar tamanho da figura automaticamente
    if figsize is None:
        figsize = (5 * n_groups, 5)
    
    fig, axes = plt.subplots(1, n_groups, figsize=figsize, sharey=True)
    if n_groups == 1:
        axes = [axes]
    
    for idx, group in enumerate(groups):
        group_data = data[data[group_column] == group][value_column]
        color = colors[idx % len(colors)]
        label = group_names[idx] if group_names and idx < len(group_names) else str(group)
        
        if len(group_data) > 1:
            # KDE plot
            sns.kdeplot(group_data, ax=axes[idx], color=color, 
                        label='Estimado', linewidth=3)
            
            mean_val = group_data.mean()
            std_val = group_data.std()
            
            # Média
            axes[idx].axvline(mean_val, color='red', linestyle='--', 
                             linewidth=3, label=f'Média = {mean_val:.3f}')
            # ±1 desvio padrão
            axes[idx].axvline(mean_val - std_val, color='red', linestyle=':', 
                             linewidth=2, alpha=0.5)
            axes[idx].axvline(mean_val + std_val, color='red', linestyle=':', 
                             linewidth=2, alpha=0.5)
            
            # Anotações
            axes[idx].text(0.05, 0.95, f'n = {len(group_data)}', 
                          transform=axes[idx].transAxes, fontsize=10, 
                          verticalalignment='top')
            axes[idx].text(0.05, 0.85, f'μ = {mean_val:.3f}', 
                          transform=axes[idx].transAxes, fontsize=10, 
                          verticalalignment='top')
            axes[idx].text(0.05, 0.75, f'σ = {std_val:.3f}', 
                          transform=axes[idx].transAxes, fontsize=10, 
                          verticalalignment='top')
        
        axes[idx].set_xlabel('Valor', fontsize=12)
        axes[idx].set_ylabel('Densidade' if idx == 0 else '', fontsize=12)
        axes[idx].set_title(label, fontsize=14)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()