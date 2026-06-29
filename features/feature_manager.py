# features/feature_manager.py
"""
Gerenciador de features para SHM - permite carregar features pré-calculadas
e integrar com o pipeline de ML
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

class FeatureManager:
    """
    Gerencia features extraídas para SHM, permitindo:
    - Carregar features de arquivos (Excel, CSV, Parquet)
    - Selecionar features específicas
    - Combinar features de diferentes fontes
    - Salvar features processadas
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame com features já extraídas
        """
        self.data = data
        self.feature_columns = []
        self.metadata = {}
        
        if data is not None:
            self.feature_columns = list(data.columns)
    
    @classmethod
    def from_excel(cls, path: str, sheet_name: str = 'DI_FRAC'):
        """Carrega features de um arquivo Excel"""
        df = pd.read_excel(path, sheet_name=sheet_name)
        return cls(df)
    
    @classmethod
    def from_csv(cls, path: str):
        """Carrega features de um arquivo CSV"""
        df = pd.read_csv(path)
        return cls(df)
    
    @classmethod
    def from_parquet(cls, path: str):
        """Carrega features de um arquivo Parquet"""
        df = pd.read_parquet(path)
        return cls(df)
    
    def select_features(self, feature_list: List[str]) -> 'FeatureManager':
        """
        Seleciona um subconjunto de features
        
        Parameters
        ----------
        feature_list : List[str]
            Lista de nomes das features a serem selecionadas
        
        Returns
        -------
        FeatureManager
            Nova instância com as features selecionadas
        """
        if self.data is None:
            raise ValueError("Nenhum dado carregado")
        
        selected_data = self.data[feature_list].copy()
        return FeatureManager(selected_data)
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Agrupa features por tipo (temporal, espectral, FRF, etc.)
        """
        if self.data is None:
            return {}
        
        groups = {
            'time': [],
            'spectral': [],
            'frf': [],
            'statistical': [],
            'domain': []
        }
        
        # Padrões para identificar tipos de features
        time_patterns = ['rms', 'mean', 'std', 'variance', 'skewness', 'kurtosis', 
                        'peak', 'crest_factor', 'shape_factor']
        spectral_patterns = ['spectral_', 'dominant_frequency', 'band_ratio']
        frf_patterns = ['frf_', 'mode_', 'peak_frequency', 'modal_']
        
        for col in self.feature_columns:
            col_lower = col.lower()
            
            if any(p in col_lower for p in frf_patterns):
                groups['frf'].append(col)
            elif any(p in col_lower for p in spectral_patterns):
                groups['spectral'].append(col)
            elif any(p in col_lower for p in time_patterns):
                groups['time'].append(col)
            else:
                groups['statistical'].append(col)
        
        # Remover grupos vazios
        return {k: v for k, v in groups.items() if v}
    
    def add_derived_features(self) -> 'FeatureManager':
        """
        Adiciona features derivadas das existentes
        Ex: razões entre features, interações, etc.
        """
        if self.data is None:
            raise ValueError("Nenhum dado carregado")
        
        df = self.data.copy()
        
        # Exemplo: adicionar razão DI-1/DI-2 se existirem
        if 'DI-1' in df.columns and 'DI-2' in df.columns:
            df['ratio_DI1_DI2'] = df['DI-1'] / (df['DI-2'] + 1e-12)
            df['diff_DI1_DI2'] = df['DI-1'] - df['DI-2']
            df['product_DI1_DI2'] = df['DI-1'] * df['DI-2']
        
        # Adicionar features polinomiais básicas
        if 'DI-1' in df.columns:
            df['DI1_squared'] = df['DI-1'] ** 2
        
        if 'DI-2' in df.columns:
            df['DI2_squared'] = df['DI-2'] ** 2
        
        return FeatureManager(df)
    
    def save(self, path: str, format: str = 'parquet'):
        """
        Salva as features em disco
        
        Parameters
        ----------
        path : str
            Caminho do arquivo
        format : str
            Formato: 'parquet', 'csv', 'excel'
        """
        if self.data is None:
            raise ValueError("Nenhum dado para salvar")
        
        if format == 'parquet':
            self.data.to_parquet(path, index=False)
        elif format == 'csv':
            self.data.to_csv(path, index=False)
        elif format == 'excel':
            self.data.to_excel(path, index=False)
        else:
            raise ValueError(f"Formato '{format}' não suportado")
        
        print(f"✅ Features salvas em: {path}")
    
    def describe(self) -> pd.DataFrame:
        """Retorna estatísticas descritivas das features"""
        if self.data is None:
            return pd.DataFrame()
        return self.data.describe()
    
    def get_feature_metadata(self) -> Dict:
        """
        Retorna metadados sobre as features
        """
        if self.data is None:
            return {}
        
        return {
            'n_features': len(self.feature_columns),
            'n_samples': len(self.data),
            'feature_names': self.feature_columns,
            'data_types': self.data.dtypes.astype(str).to_dict(),
            'has_missing': self.data.isnull().any().to_dict(),
            'feature_groups': self.get_feature_groups()
        }
    
    def to_array(self) -> np.ndarray:
        """Retorna features como array numpy"""
        if self.data is None:
            return np.array([])
        return self.data.values
    
    def get_target(self, target_column: str) -> np.ndarray:
        """Retorna a coluna alvo"""
        if self.data is None:
            return np.array([])
        if target_column not in self.data.columns:
            raise ValueError(f"Coluna '{target_column}' não encontrada")
        return self.data[target_column].values
    
    def split(self, target_column: str, test_size: float = 0.3, 
              random_state: int = 42) -> tuple:
        """
        Divide dados em treino e teste
        
        Returns
        -------
        tuple: (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X = self.data.drop(columns=[target_column]).values
        y = self.data[target_column].values
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)