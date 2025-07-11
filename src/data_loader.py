import os
import pandas as pd
import pyarrow.parquet as pq

def load_data(data_dir='data'):
    """Carga datos desde la carpeta 'data' con paths relativos seguros"""
    try:
        # Verifica existencia de archivos
        train_csv_path = os.path.join(data_dir, 'train.csv')
        test_csv_path = os.path.join(data_dir, 'test.csv')
        train_parquet_path = os.path.join(data_dir, 'train.parquet')
        test_parquet_path = os.path.join(data_dir, 'test.parquet')
        
        if not all(os.path.exists(f) for f in [train_csv_path, test_csv_path]):
            raise FileNotFoundError("Archivos CSV no encontrados en la carpeta 'data'")
        
        # Carga CSV
        train_csv = pd.read_csv(train_csv_path)
        test_csv = pd.read_csv(test_csv_path)
        
        # Carga Parquet si existen
        train_parquet = pd.DataFrame()
        test_parquet = pd.DataFrame()
        if os.path.exists(train_parquet_path):
            train_parquet = pq.read_table(train_parquet_path).to_pandas()
        if os.path.exists(test_parquet_path):
            test_parquet = pq.read_table(test_parquet_path).to_pandas()
        
        # Combina datos
        train = pd.merge(train_csv, train_parquet, on='Subject_ID', how='left') if not train_parquet.empty else train_csv
        test = pd.merge(test_csv, test_parquet, on='Subject_ID', how='left') if not test_parquet.empty else test_csv
        
        return train, test, {}
    
    except Exception as e:
        print(f"Error cargando datos: {str(e)}")
        raise