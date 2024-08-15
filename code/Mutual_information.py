import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.metrics import mutual_info_score

# Load data
tiva = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')

# Split data into train and test sets
X = tiva.drop(['class'], axis=1).reset_index(drop=True)
y = pd.DataFrame([0 if x == True else 1 for x in list(tiva['class'])], columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate mutual information
tiva_col_name = list(X_train)
mutual_info_list = []
for col_name in tiva_col_name:
    mutual_info = mutual_info_score(X_train[col_name], y_train['class'])
    mutual_info_list.append({'col_name': col_name, 'mutual_info': mutual_info})

# Convert to DataFrame
mutual_df = pd.DataFrame(mutual_info_list, columns=['col_name', 'mutual_info'])
