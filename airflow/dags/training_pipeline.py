from airflow.decorators import dag,task
from datetime import datetime

@dag(dag_id='pipeline',start_date=datetime(2020,1,1),schedule=None,catchup=False)
def training_pipeline():
    import sys
    sys.path.insert(0, "/opt/airflow")
    @task(task_id="load_data")
    def load_data():
        from src.psyche_track.data_loader import load_data,train_test_split
        import pyarrow as pa
        import pyarrow.parquet as pq
        X,y = load_data()
        X_train,X_test,y_train,y_test = train_test_split(X,y)

        X_train.to_parquet("/opt/airflow/data/X_train.parquet")
        X_test.to_parquet("/opt/airflow/data/X_test.parquet")
        y_train.to_frame().to_parquet("/opt/airflow/data/y_train.parquet")
        y_test.to_frame().to_parquet("/opt/airflow/data/y_test.parquet")
        return "data"

    @task(task_id="train_models")
    def train_models(data_path="data",with_grid_search=False):
        from src.psyche_track.pipeline import pipeline
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        def read_features(path):
            x = pd.read_parquet(path)
            return x.where(x.notna(), np.nan)

        X_train = read_features(f"/opt/airflow/{data_path}/X_train.parquet")
        X_test = read_features(f"/opt/airflow/{data_path}/X_test.parquet")
        y_train = pd.read_parquet(f"/opt/airflow/{data_path}/y_train.parquet").squeeze()
        y_test = pd.read_parquet(f"/opt/airflow/{data_path}/y_test.parquet").squeeze()

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        
        return pipeline(X_train,X_test,y_train,y_test,with_grid_search=with_grid_search)
    
    @task(task_id="compare_models")
    def compare_models(base_metrics,best_metrics):
        from src.psyche_track.evaluate import compare_models_using_metrics
        return compare_models_using_metrics(base_metrics,best_metrics)
    
    @task(task_id="choise_best_model")
    def choise_best_model(results):
        from src.psyche_track.evaluate import choice_best_model_for_streamlit
        return choice_best_model_for_streamlit(results)
    

    data_path = load_data()
    base_metrics = train_models(data_path)
    best_metrics = train_models(data_path,with_grid_search=True)
    results = compare_models(base_metrics,best_metrics)
    choise_best_model(results)


training_pipeline()