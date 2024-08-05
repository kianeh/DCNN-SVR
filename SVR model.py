# %%
"""
# 1) Load Required Dataset and Libraries
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.svm import SVC
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.metrics import classification_report

# %%
Missing_values = ["NA", "NAN", "N.A", "-", "_", " "]
df = pd.read_csv(r"C:\Users\kiane\BankChurners.csv", na_values= Missing_values) # Load Dataset from CSV File

# %%
"""
# 2) Basic Insights of Dataset
"""

# %%
print("Dataset Shape:",df.shape) # Prints Dataset Shape
print(df.info()) # Prints dtype of each column in Dataset

# %%
df.describe(include = "all") # Prints statistical summary for all columns (even Object Types)

# %%
print("Total Number of Missing Values in Dataset is:", df.isna().sum().sum()) #Prints Total Number of Missing Values in Dataset
print("Total Number of Duplicated Rows in Dataset is:", df.duplicated().sum()) #Prints Total Number of duplicate rows in Dataset

# %%
"""
# 3) Exploratory Data Analysis(EDA)
"""

# %%
df = df[['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
          'Dependent_count', 'Education_Level', 'Marital_Status',
          'Income_Category', 'Card_Category', 'Months_on_book',
          'Total_Relationship_Count', 'Months_Inactive_12_mon',
          'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
          'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
          'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]

# %%
fig, ax = plt.subplots(figsize=(10,5))
plt.rcParams.update({'font.size': 10})

fields_correlation = sns.heatmap(df.corr(), vmin=-1, cmap="PuBu", annot=True, ax=ax)

# %%
df = df.drop(['CLIENTNUM', 'Avg_Utilization_Ratio', 'Months_on_book', 'Avg_Open_To_Buy', 'Total_Trans_Amt'], axis=1) # high correlation with other features

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = df[['Customer_Age','Dependent_count','Total_Relationship_Count','Months_Inactive_12_mon'
                         ,'Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal'
                         ,'Total_Amt_Chng_Q4_Q1','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
vif

# %%
data_f = df.copy()
data_f = data_f.drop(['Customer_Age'], axis=1)
data_f = data_f.drop(['Total_Amt_Chng_Q4_Q1'], axis=1)
data_f = data_f.drop(['Total_Ct_Chng_Q4_Q1'], axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_f[['Dependent_count','Total_Relationship_Count','Months_Inactive_12_mon'
                         ,'Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Total_Trans_Ct']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
vif

# %%
data_f.drop(['Card_Category'], axis=1)

# %%
data_f.shape

# %%
"""
# 4) SVM Classification Model
"""

# %%
"""
## 4-1) Train and Test Split
"""

# %%
y_dict = {"Existing Customer": 1,"Attrited Customer": 0}
data_f["Attrition_Flag"]  = [y_dict[g] for g in data_f["Attrition_Flag"]]
X = data_f.drop(["Attrition_Flag"], axis=1) # Independent Variables
Y = data_f["Attrition_Flag"]                             # Dependent Variable (target Variable)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=2) # Split Dataset to Train (75%) and Test (25%) Datasdet
print("Before OverSampling, counts of label 'Existing Customer' in Train Dataset: {}".format(sum(y_train == 0)))
print("Before OverSampling, counts of label 'Attrited Customer' in Train Dataset: {} \n".format(sum(y_train == 1)))

# %%
"""
## 4-2) Define PipeLines
"""

# %%
# Creating a pair of Pipeline objects, one each for the numeric and categorical transformations

num_transformer = Pipeline(steps=[('scaler', StandardScaler())]) # Scale Numerical values
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) # One-hot encode categorical values

# Using ColumnTransformers to Select different column Types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, selector(dtype_include=['float64', 'int64', 'int'])),
        ('cat', cat_transformer, selector(dtype_include=['object']))])

my_pipeline = imbpipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('smote', SMOTE(random_state=42)), # Because the Dataset is unbalanced Use SMOTE for Oversampling and making the Train Dataset balanced                                                                                          
                                ('regressor', SVR())
                                ])

# %%
"""
## 4-3) Define GridSearchCV to Choose Best Parameters
"""

# %%
# You can test different parameters here, for better performance i commented out lots of possible parameters.
GridSearch_parameters = [
    {'regressor' : [SVR()],
    'preprocessor__num__scaler': [StandardScaler()],
    'regressor__max_iter': [1000],
    'regressor__kernel': ['rbf','linear','poly'],
    },
]
stratified_kfold = StratifiedKFold(n_splits=5,
                                    shuffle=True,
                                    random_state=42)

# %%
# Using GridSearchCV on your pipeline with the number of folds to find the best parameters
grid_search= GridSearchCV(estimator=my_pipeline,
                            param_grid=GridSearch_parameters,
                            cv=stratified_kfold, # 5 fold cross validation
                            scoring='roc_auc',
                            n_jobs=-1,
                            )
grid_search.fit(x_train, y_train) #fit the pipeline to train dataset

# %%
# Best results is with StandardScaler and model kernel as 'rbf', with gamma 0.1 and C= 5
print(grid_search.best_params_) # to show the best parameters


# %%
"""
## 4-4) Performance Evaluation of the Regression Model
"""

# %%
test_score = grid_search.score(x_test, y_test)
print(f'Cross-validation score: {grid_search.best_score_}\nTest score: {test_score}')
