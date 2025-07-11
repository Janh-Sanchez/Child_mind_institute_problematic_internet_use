import os
from pathlib import Path
import pandas as pd
from features import FeatureProcessor
from modtrain import train_and_evaluate
from utils import save_submission

# Configuración robusta de paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'

def main():
    try:
        print("\n=== Cargando datos ===")
        train = pd.read_csv(DATA_DIR / 'train.csv')
        test = pd.read_csv(DATA_DIR / 'test.csv')
        
        # Verificación de datos críticos
        assert 'PCIAT-PCIAT_Total' in train.columns, "Columna target no encontrada"
        train = train.dropna(subset=['PCIAT-PCIAT_Total'])
        
        print("\n=== Preprocesamiento ===")
        processor = FeatureProcessor()
        train, test = processor.preprocess(train, test)
        
        print("\n=== Preparando target ===")
        # Versión robusta de qcut
        train['SII_group'], bins = pd.qcut(
            train['PCIAT-PCIAT_Total'],
            q=4,
            labels=[0, 1, 2, 3],
            retbins=True,
            duplicates='drop'
        )
        y = train['SII_group'].astype(int)
        
        print("\n=== Seleccionando features ===")
        # Excluir columnas no relevantes y asegurar consistencia
        exclude = ['PCIAT-PCIAT_Total', 'Subject_ID', 'SII_group', 'timestamp', 'id']
        
        # Solo features presentes en ambos datasets
        common_features = list(set(train.columns) & set(test.columns))
        features = [
            col for col in common_features
            if col not in exclude
            and pd.api.types.is_numeric_dtype(train[col])
            and col in test.columns
        ]
        
        print(f"Features seleccionadas: {len(features)}")
        X = train[features]
        
        print("\n=== Entrenamiento ===")
        model, qwk = train_and_evaluate(X, y)
        print(f"\n✔ QWK promedio: {qwk:.4f}")
        
        print("\n=== Generando submission ===")
        # Verificar features en test
        missing_in_test = [col for col in features if col not in test.columns]
        if missing_in_test:
            print(f"⚠ Features faltantes en test: {missing_in_test}")
            features = [col for col in features if col in test.columns]
        
        test_preds = model.predict(test[features])
        
        # Asegurar ID para submission
        if 'id' not in test.columns and 'Subject_ID' in test.columns:
            test['id'] = test['Subject_ID']
        elif 'id' not in test.columns:
            test['id'] = range(len(test))
        
        save_submission(test, test_preds, OUTPUT_DIR)
        print(f"✔ Submission generado en {OUTPUT_DIR / 'submission.csv'}")
        
    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        raise

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()