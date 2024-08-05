#!/usr/bin/env python
# coding: utf-8

# In[42]:


from PIL import Image
import numpy as np
import numpy as nppip
import matplotlib.pyplot as plt
import os
import time

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical


# In[3]:


df = pd.read_csv(r"C:\Users\kiane\BankChurners.csv")


# In[4]:


df


# In[128]:


Missing_values = ["NA", "NAN", "N.A", "-", "_", " "]
df = pd.read_csv(r"C:\Users\kiane\BankChurners.csv", na_values= Missing_values) # Load Dataset from CSV File

# %% [markdown]
# # 2) Basic Insights of Dataset

# %%
print("Dataset Shape:",df.shape) # Prints Dataset Shape
#print(df.info()) # Prints dtype of each column in Dataset

# %%
df.describe(include = "all") # Prints statistical summary for all columns (even Object Types)

# %%
print("Total Number of Missing Values in Dataset is:", df.isna().sum().sum()) #Prints Total Number of Missing Values in Dataset
print("Total Number of Duplicated Rows in Dataset is:", df.duplicated().sum()) #Prints Total Number of duplicate rows in Dataset

# %% [markdown]
# # 3) Exploratory Data Analysis(EDA)

# %%
df = df[['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
          'Dependent_count', 'Education_Level', 'Marital_Status',
          'Income_Category', 'Card_Category', 'Months_on_book',
          'Total_Relationship_Count', 'Months_Inactive_12_mon',
          'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
          'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
          'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]

# %%
g_unq = df['Gender'].unique()
g_map = {g_unq[i]:i for i in range(len(g_unq))}
df['Gender'] = df['Gender'].map(g_map)

el_unq = df['Education_Level'].unique()
el_map = {el_unq[i]:i for i in range(len(el_unq))}
#print("elmap   ", el_map)
df['Education_Level'] = df['Education_Level'].map(el_map)

cc_unq = df['Card_Category'].unique()
cc_map = {cc_unq[i]:i for i in range(len(cc_unq))}
df['Card_Category'] = df['Card_Category'].map(cc_map)

ic_unq = df['Income_Category'].unique()
ic_map = {ic_unq[i]:i for i in range(len(ic_unq))}
df['Income_Category'] = df['Income_Category'].map(ic_map)

ms_unq = df['Marital_Status'].unique()
ms_map = {ms_unq[i]:i for i in range(len(ms_unq))}
df['Marital_Status'] = df['Marital_Status'].map(ms_map)

af_unq = df['Attrition_Flag'].unique()
af_map = {af_unq[i]:i for i in range(len(af_unq))}
df['Attrition_Flag'] = df['Attrition_Flag'].map(af_map)



X = df.drop(["CLIENTNUM", "Attrition_Flag"], axis=1) # Independent Variables
Y = df["Attrition_Flag"]                             # Dependent Variable (target Variable)
print("Before OverSampling, counts of label 'Existing Customer' in Train Dataset: {}".format(sum(Y == "Existing Customer")))
print("Before OverSampling, counts of label 'Attrited Customer' in Train Dataset: {} \n".format(sum(Y == "Attrited Customer")))


sm = SMOTE(random_state=2)
X_res, y_res = sm.fit_resample(X, Y)

print("After OverSampling, counts of label 'Existing Customer' in Train Dataset: {}".format(sum(y_res == "Existing Customer")))
print("After OverSampling, counts of label 'Attrited Customer' in Train Dataset: {} \n".format(sum(y_res == "Attrited Customer")))

x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, stratify=y_res, random_state=2)

y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values


# In[129]:


# x_train = x_train[:,:,np.newaxis]
# x_test = x_test[:,:,np.newaxis]

x_train = x_train[:,np.newaxis,:]
x_test = x_test[:,np.newaxis,:]


# In[131]:


print(x_train.shape)

print(y_train_oh.shape)
print(x_test.shape)


# In[ ]:





# In[158]:


model =keras.models.Sequential([
    #keras.layers.Conv1D(20, 10, activation="relu",padding="same", input_shape=(19,1)),
    keras.layers.Conv1D(20, 1, activation="relu",padding="same", input_shape=(1,19)),
    #keras.layers.Conv1D(20, 10, activation="relu",padding="same"),
    keras.layers.MaxPooling1D(1),
#     keras.layers.Conv1D(50, 5, activation="relu",padding="same"),
#     keras.layers.MaxPooling1D(2),
#     keras.layers.Conv1D(100, 3, activation="relu", padding="same"),
#     keras.layers.AveragePooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(2)                                
])


# In[159]:


model.compile(loss = 'mean_squared_error',
              optimizer="adam",
              metrics=["mean_absolute_error","accuracy"])


# In[160]:


# print(x_train.shape)
# print(y_train.shape)

history = model.fit(
    x_train,
    y_train_oh, 
    validation_data=(x_test, y_test_oh), 
    epochs=100, 
    batch_size=16,
    #callbacks=callbacks_list
)


# In[ ]:





# In[ ]:





# In[ ]:
from sklearn.metrics import f1_score, precision_score, recall_score
# After training your model

# Make predictions on the test set
y_pred = model.predict(x_test)

# Convert predicted probabilities to binary labels
y_pred_binary = np.argmax(y_pred, axis=1)
y_test_binary = np.argmax(y_test_oh, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test_binary, y_pred_binary)
recall = recall_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)

# Print precision, recall, and F1 score
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)





