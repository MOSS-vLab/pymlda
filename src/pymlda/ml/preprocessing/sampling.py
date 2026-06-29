# SUBSTITUIR TODO O CONTEÚDO DE sampling.py:

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt

class UnderSampler:
    """Balanceamento de dados por undersampling."""
    
    def __init__(self, random_state=42):
        self.rus = RandomUnderSampler(random_state=random_state)
        self.random_state = random_state
    
    def fit_resample(self, X, y):
        """Aplica undersampling para balancear as classes."""
        return self.rus.fit_resample(X, y)
    
    def get_sampling_strategy(self, y):
        """Retorna a estratégia de amostragem."""
        counter = Counter(y)
        n_samples = min(counter.values())
        return {cls: n_samples for cls in counter.keys()}
    
    def plot_balance(self, y_before, y_after):
        """Plota a distribuição das classes antes e depois."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        before = Counter(y_before)
        after = Counter(y_after)
        
        axes[0].bar(before.keys(), before.values(), color='red', alpha=0.7)
        axes[0].set_title('Antes do Undersampling')
        axes[0].set_xlabel('Classe')
        axes[0].set_ylabel('Contagem')
        
        axes[1].bar(after.keys(), after.values(), color='green', alpha=0.7)
        axes[1].set_title('Depois do Undersampling')
        axes[1].set_xlabel('Classe')
        axes[1].set_ylabel('Contagem')
        
        plt.tight_layout()
        plt.show()