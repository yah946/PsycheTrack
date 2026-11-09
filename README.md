# 🧠 PsycheTrack: Student Mental Health Risk Analysis & Prediction

**PsycheTrack** is an educational Data Science and Machine Learning project focused on analyzing student mental health-related data, identifying factors associated with psychological well-being, and developing a supervised classification prototype to predict risk levels.

The project covers the complete Data Science lifecycle, from data exploration and preparation to model optimization, evaluation, workflow orchestration, and interactive visualization.

> **Disclaimer:** PsycheTrack is strictly an educational project. Its predictions are experimental and must never be used for clinical diagnosis, psychological screening, disciplinary decisions, student selection, or any other high-stakes decision. The model is not a substitute for qualified mental health professionals.

---

## 🎯 Project Objectives

The project aims to explore student-related factors associated with mental health risks and build a supervised learning prototype.

Main objectives:

- **Risk Analysis:** Explore patterns associated with different mental health risk levels.
- **Prevention:** Understand potential indicators that may help inform student support initiatives.
- **Factor Analysis:** Identify relationships between student characteristics, lifestyle factors, and risk levels.
- **Student Support:** Explore how data analysis could contribute to better-informed support strategies.
- **Predictive Modeling:** Develop and evaluate machine learning classification models.

---

## 📊 Dataset

The project uses the **Student Mental Health Risk Prediction Dataset**, available on Kaggle.

**Source:** [Mental Health Risk Prediction Dataset - Students](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)

The dataset contains student-related information intended for analyzing factors associated with mental health risk.

Before modeling, the dataset will be audited to understand its structure, data quality, variables, and target distribution.

---

## 🔬 Project Workflow

The project follows five main stages.

### 1. Data Loading & Exploratory Data Analysis (EDA)

The first stage focuses on understanding the dataset and identifying potential data quality issues.

Tasks:

- Load and inspect the dataset.
- Analyze dimensions, column names, and data types.
- Identify missing values and duplicate records.
- Detect inconsistencies and potential outliers.
- Generate descriptive statistics.
- Analyze the target variable distribution and class imbalance.
- Explore relationships between explanatory variables and the target.
- Examine correlations between numerical variables.
- Identify variables associated with different risk levels.

**Expected outputs:**
- Initial dataset overview.
- Data quality assessment.
- Descriptive statistics.
- EDA visualizations.
- Preliminary findings and observations.

### 2. Data Cleaning & Preprocessing

Prepare the dataset for machine learning while ensuring that every transformation is justified.

Tasks:

- Split the dataset into training (80%) and testing (20%) sets.
- Handle missing values using appropriate imputation strategies.
- Detect and investigate outliers using methods such as IQR or Z-score.
- Encode categorical variables.
- Handle class imbalance using techniques such as SMOTE or undersampling, if necessary.
- Scale numerical features using normalization or standardization when appropriate.
- Build preprocessing pipelines using Scikit-learn.
- Prevent data leakage by fitting transformations exclusively on training data.

**Important:** Each cleaning and preprocessing decision must be supported by an explanation of its statistical and practical relevance.

**Expected outputs:**
- Cleaned dataset.
- Preprocessing pipeline.
- Documented transformation decisions.

### 3. Model Training & Comparison

Train and compare multiple supervised classification algorithms using the same training dataset.

Tasks:

- Select at least three classification algorithms.
- Train the selected models.
- Compare their predictive performance.
- Evaluate models using appropriate classification metrics.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (when appropriate)

**Expected outputs:**
- Trained classification models.
- Model comparison results.
- Initial performance visualizations.

### 4. Hyperparameter Optimization & Model Evaluation

Improve promising models and evaluate their generalization performance.

Tasks:

- Select the most promising models.
- Apply GridSearchCV or RandomizedSearchCV.
- Optimize model hyperparameters.
- Compare optimized models.
- Select a model based on appropriate evaluation criteria.
- Evaluate the final model on the independent test set.
- Analyze false positives and false negatives.
- Discuss model limitations, potential biases, and generalization issues.
- Document and justify the final model selection.
- Save the complete preprocessing and prediction pipeline.

**Expected outputs:**
- Optimized model.
- Final evaluation report.
- Confusion matrix.
- Error analysis.
- Serialized pipeline (`.pkl` or `.joblib`).

### 5. Workflow Orchestration & Interactive Application

Automate the machine learning workflow and provide an interface for exploring the results.

**Apache Airflow**

Develop an Airflow DAG to orchestrate the machine learning pipeline.

The workflow will cover:

- Data preparation.
- Data preprocessing.
- Model training.
- Model evaluation.
- Model artifact storage.

Additional objectives:

- Configure scheduled DAG execution.
- Monitor task execution and status.
- Handle and investigate pipeline failures.
- Ensure reproducibility.

**Streamlit**

Develop an interactive application to:

- Display model evaluation metrics.
- Visualize model performance.
- Explore confusion matrices.
- Compare classification models.
- Test the saved pipeline with new input data.
- Display prediction results with appropriate educational disclaimers.

**Expected outputs:**
- Automated Airflow DAG.
- Interactive Streamlit application.
- Reusable machine learning pipeline.

---

## 🛠️ Technologies & Tools

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Package & Project Manager | uv |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Imbalanced Data | Imbalanced-learn |
| Model Optimization | GridSearchCV, RandomizedSearchCV |
| Model Serialization | Joblib |
| Workflow Orchestration | Apache Airflow |
| Interactive Dashboard | Streamlit |
| Environment | Jupyter Notebook, Docker |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

The project is organized into separate directories to maintain a clear distinction between data, experimentation, source code, and generated artifacts.

```text
PsycheTrack/
│
├── data/
│   ├─- raw/                  # Original dataset
│   ├── processed/            # Cleaned and transformed data
│   └── external/             # Additional datasets, if needed
│
├── notebooks/
│   ├── 01_exploration.ipynb  # Data inspection and EDA
│   ├── 02_preprocessing.ipynb# Cleaning and feature engineering
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data/                 # Data loading and validation
│   ├── features/             # Feature engineering and preprocessing
│   ├── models/               # Training and evaluation
│   └── visualization/        # Visualization utilities
│
├── airflow/
│   └── dags/                 # Airflow DAG definitions
│
├── app/
│   └── streamlit_app.py      # Interactive application
│
├── models/                   # Saved model pipelines
│
├── reports/
│   ├── figures/              # Generated visualizations
│   └── metrics/              # Evaluation reports
│
├── tests/                    # Unit and pipeline tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

*The structure may evolve as the project develops.*

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yah946/PsycheTrack
cd PsycheTrack
```

### 2. Create a virtual environment

```bash
uv venv
```

Activate it:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv add pandas numpy matplotlib seaborn scikit-learn
```

### 4. Download the dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students) and place the original file inside:

```text
data/raw/
```

### 5. Explore the project

Start with the first notebook:

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

---

## 📈 Evaluation Strategy

Model selection will not rely exclusively on accuracy.

Because the project concerns mental health-related risk categories, it is important to examine how different classification errors affect the interpretation of model performance.

The evaluation will therefore consider:

- Precision and recall for each class.
- Macro and weighted F1-scores.
- Confusion matrices.
- ROC-AUC when applicable.
- Class imbalance.
- Potential bias and limitations.

The final model will be selected based on its overall performance, reliability, and limitations rather than a single metric.

---

## ⚖️ Ethical Considerations & Limitations

Mental health is a sensitive subject, and predictive models in this domain require particular caution.

PsycheTrack follows these principles:

- Predictions are experimental and educational only.
- The model does not diagnose psychological or psychiatric conditions.
- Predictions must not be used to label, rank, punish, or exclude students.
- Correlations must not be interpreted as causal relationships.
- Dataset quality, representativeness, and potential biases must be investigated.
- Model uncertainty and limitations must be communicated clearly.
- Any real-world student support must involve appropriate human professionals and institutional safeguards.

---

## 🎓 Learning Outcomes

Through this project, the goal is to develop practical and theoretical skills in:

- Exploratory Data Analysis.
- Data Quality Assessment.
- Data Cleaning and Feature Engineering.
- Statistical Analysis and Visualization.
- Supervised Machine Learning.
- Classification Model Evaluation.
- Hyperparameter Optimization.
- Scikit-learn Pipelines.
- Data Leakage Prevention.
- Machine Learning Workflow Automation.
- Interactive Data Applications.

---

## 📌 Project Status

<mark>In Development</mark>

The project will be developed progressively, following the complete Data Science workflow, with each stage documented and validated before moving to the next one.

---

**PsycheTrack - Exploring Student Well-being Through Data Science.**
