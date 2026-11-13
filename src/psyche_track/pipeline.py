import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from pathlib import Path
from .data_loader import (
    load_data,
    train_test_split,
    preprocessing
)
import joblib
RANDON_SEED = 10
COPARAISON_METRICS_PATH = Path('models/compare/metrics')
COPARAISON_MODEL_PATH = Path('models/compare/metrics')
MODELS_PATH = Path('models')
models_metrics = []
MODELS = {
    "Logistic Regression": {
        "classifier": sklearn.linear_model.LogisticRegression(random_state=RANDON_SEED),
        "params": {
            'classifier__max_iter': [1000, 2000, 3000],
            'classifier__penalty': ['l1','l2'],
            'classifier__solver': ['saga'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        },
        "path": MODELS_PATH/'logistic_regression'
    },
    "K-Nearest Neighbors": {
        "classifier": sklearn.neighbors.KNeighborsClassifier(),
        "params": {
            'classifier__n_neighbors': [1, 5, 7, 9, 10],
            'classifier__p': [1, 2],
            'classifier__leaf_size': [20, 50, 100],
            'classifier__weights': ['uniform', 'distance'],
        },
        "path": MODELS_PATH/'knn'
    },
    "Support Vector Classifier": {
        "classifier": sklearn.svm.SVC(probability=True,random_state=RANDON_SEED),
        "params": {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        },
        "path": MODELS_PATH/'svc'
    }
}
def train_model(classifier,preprocessor,param_grid,X_train,y_train,with_grid_search=False):
    pipeline = sklearn.pipeline.Pipeline([('preprocess',preprocessor),('classifier',classifier)])
    if with_grid_search:
        grid_search = sklearn.model_selection.GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=10),scoring='f1_weighted',n_jobs=-1,verbose=1,error_score='raise')
        grid_search.fit(X_train,y_train)
        model = grid_search.best_estimator_
    else:
        model = pipeline.fit(X_train,y_train)
    return model
def evaluate_model(y_test,y_pred,y_proba,average='macro',multi_class='ovr',decimal_point=2):
    accuracy = round(sklearn.metrics.accuracy_score(y_test,y_pred)*100,decimal_point)
    precision = round(sklearn.metrics.precision_score(y_test,y_pred,average=average)*100,decimal_point)
    recall = round(sklearn.metrics.recall_score(y_test,y_pred,average=average)*100,decimal_point)
    f1 = round(sklearn.metrics.f1_score(y_test,y_pred,average=average)*100,decimal_point)
    auc = round(sklearn.metrics.roc_auc_score(y_test,y_proba,multi_class=multi_class,average=average),decimal_point*2)
    return accuracy,precision,recall,f1,auc
def save_cm(name,y_test,y_pred,classes,metric_path):
    cm = sklearn.metrics.confusion_matrix(y_test,y_pred,labels=classes)
    sns.heatmap(cm,annot=True,xticklabels=classes,yticklabels=classes)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predict')
    plt.ylabel('Real')
    plt.savefig(metric_path/'confusion_matrix.png',dpi=300, bbox_inches='tight')
    plt.close()
def save_roc_curve(name,y_test_bin,y_proba,classes,metric_path,figsize=(8,6)):
    _,ax = plt.subplots(figsize=(8,6))
    for i, class_name in enumerate(classes):
        sklearn.metrics.RocCurveDisplay.from_predictions(y_test_bin[:, i],y_proba[:, i],name=f"Class {name}",ax=ax)
    plt.title("Multiclass ROC Curve (OvR)")
    plt.savefig(metric_path/'roc_curve.png',dpi=300, bbox_inches='tight')
    plt.close()
    
def pipeline(X_train,X_test,y_train,y_test,models:dict =MODELS,with_grid_search=False):
    stage = 'fine_tuning' if with_grid_search else 'base'
    preprocessor = preprocessing()
    for name,config in models.items():
        metric_path = config['path']/stage/'metrics'
        model_path = config['path']/stage/'model'
        os.makedirs(metric_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        model = train_model(config['classifier'],preprocessor,config['params'],X_train,y_train,with_grid_search)
        classes = model.classes_
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        y_test_bin = sklearn.preprocessing.label_binarize(y_test, classes=classes)
        accuracy,precision,recall,f1,auc = evaluate_model(y_test,y_pred,y_proba)
        models_metrics.append({'name':name,'stage':stage,'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'auc':auc})
        save_cm(name,y_test,y_pred,classes,metric_path)
        save_roc_curve(name,y_test_bin,y_proba,classes,metric_path)
        model_joblib_name = 'best_model.joblib' if with_grid_search else 'base_model.joblib'
        joblib.dump(model,model_path/model_joblib_name)
    return models_metrics

    

    
    