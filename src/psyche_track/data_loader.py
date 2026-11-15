from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

RANDON_SEED = 10
IMPORT_DATASET_PATH = Path('data/raw')
EXPORT_DATASET_PATH = Path('data/processed')
independent_numerical_columns = ['Year_of_Study','Study_Hours_Per_Week','GPA','Social_Media_Hours','Screen_Time_Hours']
related_numerical_columns = ['Age','Sleep_Hours','Financial_Stress','Family_Support','Attendance_Rate']
ordinal_columns = {'Diet_Quality':['Poor','Average','Good'],'Physical_Activity':['Low','Moderate','High']}
nominal_columns = ['Gender','Part_Time_Job','Previous_Mental_Health_History']


def load_data(import_path:Path=IMPORT_DATASET_PATH,export_path:Path=EXPORT_DATASET_PATH,dataset_name:str='student_mental_health.csv',target:str='Mental_Health_Risk'):
    df = pd.read_csv(import_path/dataset_name)
    
    screen_time_mask = df.Social_Media_Hours > df.Screen_Time_Hours
    df = df[~screen_time_mask]
    dataset_path_name = f'cleaned_{dataset_name}'
    df.to_csv(export_path/dataset_path_name, index=False)

    X = df.drop(target,axis=1)
    y = df[target]

    return X,y

def train_test_split(X:pd.DataFrame,y:pd.Series,train_size:float=0.8):
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=train_size,random_state=RANDON_SEED,stratify=y)
    return X_train,X_test,y_train,y_test

def preprocessing(ind_num_cols:list=independent_numerical_columns,rlt_num_cols:list=related_numerical_columns,ord_cols:list=ordinal_columns,nml_cols:list=nominal_columns)->ColumnTransformer:

    independent_numerical_pipeline = Pipeline([('imputer',SimpleImputer(strategy='mean')),('scaler',sklearn.preprocessing.StandardScaler())])
    related_numerical_pipeline = Pipeline([('scaler',StandardScaler()),('imputer',KNNImputer(n_neighbors=5,weights='distance')),])
    nominal_pipeline = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))])
    ordinal_pipeline = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OrdinalEncoder(categories=[ordinal_columns['Diet_Quality'],ordinal_columns['Physical_Activity']]))])

    preprocessor = ColumnTransformer([
        ('ind_num',independent_numerical_pipeline,ind_num_cols),
        ('rlt_num',related_numerical_pipeline,rlt_num_cols),
        ('nom_cat',nominal_pipeline,nml_cols),
        ('ord_cat',ordinal_pipeline,[column for column in ord_cols])
    ])

    return preprocessor