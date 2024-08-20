# !pip install boruta
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from boruta import BorutaPy
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load the data
tiva = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')

# Separate features (X) and target (y)
X = tiva.drop(['class'], axis=1)
y = pd.DataFrame([0 if x else 1 for x in tiva['class']], columns=['class'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling
oversample = RandomOverSampler(sampling_strategy='minority', random_state=123)
X_res, y_res = oversample.fit_resample(X_train, y_train)

# Apply Boruta algorithm
boruta = BorutaPy(XGBClassifier(random_state=42), n_estimators='auto', verbose=2, random_state=42)
boruta.fit(X_res.values, y_res.values.ravel())

# Keep only the selected features
selected_cols = X_train.columns[boruta.support_]

# Create new DataFrames with selected features
X_res_select = X_res[selected_cols]
X_test_select = X_test[selected_cols]

# Train XGBoost model
xgb = XGBClassifier(random_state=42)
xgb.fit(X_res_select, y_res)

# Predict
y_pred = xgb.predict(X_test_select)

# Evaluate
F1_score_val = f1_score(y_test.values.ravel(), y_pred, average='macro')
print("F1_score:", F1_score_val)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["True: MACE", "False: not-MACE"], digits=3))
