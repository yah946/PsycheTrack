from pathlib import Path
import pandas as pd
import os
from shutil import copy2

COMPARAISON_PATH = Path('models/compare')
MODELS_PATH = Path('models')
files = {
    'K-Nearest Neighbors':'knn',
	'Logistic Regression':'logistic_regression',
	'Support Vector Classifier':'svc'
}
def compare_models_using_metrics(base_metrics,best_metrics=None,export_path=COMPARAISON_PATH):
    result = pd.DataFrame(base_metrics+best_metrics)
    result = result.sort_values('name').reset_index().drop('index',axis=1)
    metric_path = export_path/'metrics'
    os.makedirs(metric_path, exist_ok=True)
    result.to_csv(metric_path/'comparaision_between_models.csv')
    return result

def choice_best_model_for_streamlit(metrics:pd.DataFrame,export_path=COMPARAISON_PATH,models_path=MODELS_PATH):
    sorted_metrics = metrics.sort_values('f1',ascending=False).reset_index().drop('index',axis=1)
    model_name = sorted_metrics.loc[0,'name']
    model_stage = sorted_metrics.loc[0,'stage']
    file_name = files[model_name]
    os.makedirs(export_path/'model', exist_ok=True)
    distination_path = export_path/'model'/'classifier.joblib'
    source_path = next((models_path/file_name/model_stage/'model').glob('*.joblib'))  
    copy2(source_path, distination_path)
    return True