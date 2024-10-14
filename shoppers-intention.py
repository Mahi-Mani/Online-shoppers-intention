# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:34:49 2024

@author: mmahi
"""

import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import matplotlib.pyplot as plt
import graphviz
import pydotplus
from IPython.display import Image
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# READ THE DATA
df = pd.read_excel(r"D:\online_shoppers_intention(1).xlsx")
print(df)

# FEATURE ENGINEERING
# 1. CREATE TWO NEW COLUMNS
def calculate_page_values_per_product_dur(row):
    if row['ProductRelated_Duration'] != 0:
        return row['PageValues'] / row['ProductRelated_Duration']
    else:
        return np.inf 
    
def calculate_page_values_per_product_view(row):
    if row['ProductRelated_Duration'] != 0:
        return row['PageValues'] / row['ProductRelated']
    else:
        return np.inf 
    
df['page_values_per_product_dur'] = df.apply(calculate_page_values_per_product_dur, axis=1)
df['page_values_per_product_view'] = df.apply(calculate_page_values_per_product_view, axis=1)
print(df)

# 2. ENCODING FOR CATEGORICAL VARIABLES
duration_mapping = {
    'Short':0,
    'Medium':1,
    'Long':2
    }

month_mapping = {
    'Feb': 2,
    'Mar': 3,
    'May': 5,
    'June': 6,
    'Jul':7,
    'Aug':8,
    'Sep':9,
    'Oct':10,
    'Nov':11,
    'Dec':12
}

df['DurationPeriod_enc'] = df['DurationPeriod'].map(duration_mapping)
df['Month_enc'] = df['Month'].map(month_mapping)
seasons_enc = pd.get_dummies(df, columns=['Seasons'])
df = pd.concat([df, seasons_enc], axis=1)
visitor_enc = pd.get_dummies(df, columns=['VisitorType'])
df = pd.concat([df, visitor_enc], axis=1)
df = df.replace({False:0, True:1})
df_transposed = df.T
df_transposed = df_transposed.drop_duplicates()
df_cleaned = df_transposed.T
df = df_cleaned


# 3. DROP UNRELATED COLUMS
columns_to_drop = ['OperatingSystems','Browser','TrafficType']
df = df.drop(columns=columns_to_drop)

# 4. COPY TO A NEW DATASET - DROP OLD COLUMNS - SCALE NUMERIC FEATURES
dfs = pd.DataFrame()
dfs = df.copy()
columns_to_drop = ['Month','VisitorType','DurationPeriod','Seasons']
dfs = dfs.drop(columns=columns_to_drop)
dfs = dfs.replace({np.inf:-1})
columns_to_standardize = ['Administrative', 'Administrative_Duration', 'Informational','Informational_Duration',
                          'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues',
                          'SpecialDay','Region','Totalpagesvisited','TotalDuration','page_values_per_product_view',
                          'page_values_per_product_dur','Month_enc','DurationPeriod_enc']

ct = ColumnTransformer([('standardize', StandardScaler(), columns_to_standardize)], remainder='passthrough')
X_transformed = ct.fit_transform(dfs)

# 4. MODEL SVM
y = dfs['Revenue']
X = dfs.drop('Revenue', axis=1)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# hyper parameter tuning

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Define the hyperparameters to search
param_grid = {
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.1, 1, 10]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy - SVM:", accuracy)

#svm = SVC(kernel='linear', C=1.0)
#svm.fit(X_train, y_train)
#y_pred = svm.predict(X_val)
#val_accuracy = accuracy_score(y_val, y_pred)

#print("Validation Accuracy - SVM:", val_accuracy)

# Evaluate the selected model on the test set
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Test Accuracy - SVM:", test_accuracy)

### try manually
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) #88%

svm = SVC(kernel='linear', C=10)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) #88%

svm = SVC(kernel='linear', C=0.1)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) #88%

svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) # 85%

svm = SVC(kernel='rbf', C=10)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) #87%

svm = SVC(kernel='rbf', C=0.1)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) #84%

svm = SVC(kernel='poly', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) #84%

svm = SVC(kernel='poly', C=10)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) #85%

svm = SVC(kernel='poly', C=0.1)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_accuracy) #84%