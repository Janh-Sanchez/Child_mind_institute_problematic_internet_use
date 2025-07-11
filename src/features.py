import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sentence_transformers import SentenceTransformer

class FeatureProcessor:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.text_cols = []
        self.cat_cols = []
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.embed_cols = []
        
    def preprocess(self, train, test):
        """Versión optimizada que evita fragmentación"""
        try:
            # 1. Identificar columnas seguras
            self._identify_safe_columns(train, test)
            
            # 2. Procesar texto
            if self.text_cols:
                train, test = self._process_text_optimized(train, test)
            
            # 3. Codificar categóricas
            if self.cat_cols:
                train, test = self._encode_categoricals(train, test)
            
            # 4. Normalizar numéricas
            train, test = self._scale_numerical(train, test)
            
            return train, test
            
        except Exception as e:
            print(f"Error en preprocesamiento: {str(e)}")
            raise
    
    def _identify_safe_columns(self, train, test):
        """Identifica columnas existentes en ambos datasets"""
        common_cols = list(set(train.columns) & set(test.columns))
        
        self.text_cols = [
            col for col in common_cols 
            if train[col].dtype == 'object' 
            and train[col].str.contains('[a-zA-Z]', regex=True, na=False).any()
        ]
        
        self.cat_cols = [
            col for col in common_cols 
            if train[col].dtype == 'object' 
            and col not in self.text_cols
        ]
    
    def _process_text_optimized(self, train, test):
        """Procesamiento de texto sin fragmentación"""
        # Generar todos los embeddings primero
        train_embeddings = []
        test_embeddings = []
        
        for col in self.text_cols:
            train_text = train[col].fillna('').astype(str)
            test_text = test[col].fillna('').astype(str)
            
            # Embeddings para train y test
            train_emb = self.model.encode(train_text.tolist(), show_progress_bar=False)
            test_emb = self.model.encode(test_text.tolist(), show_progress_bar=False)
            
            train_embeddings.append(train_emb)
            test_embeddings.append(test_emb)
        
        # Concatenar todos los embeddings horizontalmente
        if train_embeddings:
            train_embeddings = np.hstack(train_embeddings)
            test_embeddings = np.hstack(test_embeddings)
            
            # Crear DataFrames completos antes de asignar
            n_features = train_embeddings.shape[1]
            self.embed_cols = [f'text_embed_{i}' for i in range(n_features)]
            
            train_emb_df = pd.DataFrame(train_embeddings, columns=self.embed_cols, index=train.index)
            test_emb_df = pd.DataFrame(test_embeddings, columns=self.embed_cols, index=test.index)
            
            # Concatenar de una sola vez
            train = pd.concat([train, train_emb_df], axis=1)
            test = pd.concat([test, test_emb_df], axis=1)
            
        return train, test
    
    def _encode_categoricals(self, train, test):
        """Codificación segura de categóricas"""
        if self.cat_cols:
            train_cats = train[self.cat_cols]
            test_cats = test[self.cat_cols]
            
            # Codificar y asignar de una vez
            train[self.cat_cols] = self.encoder.fit_transform(train_cats)
            test[self.cat_cols] = self.encoder.transform(test_cats)
            
        return train, test
    
    def _scale_numerical(self, train, test):
        """Normalización optimizada"""
        num_cols = [col for col in train.select_dtypes(include=np.number).columns 
                   if col not in ['Subject_ID', 'PCIAT-PCIAT_Total'] and col in test.columns]
        
        if num_cols:
            means = train[num_cols].mean()
            stds = train[num_cols].std() + 1e-8
            
            train[num_cols] = (train[num_cols] - means) / stds
            test[num_cols] = (test[num_cols] - means) / stds
            
        return train, test