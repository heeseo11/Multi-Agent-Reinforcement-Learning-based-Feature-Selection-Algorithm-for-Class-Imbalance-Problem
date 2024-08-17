import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb

# Load the data
tiva = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')

# Separate features (X) and target (y)
X = tiva.drop(['class'], axis=1).reset_index(drop=True)
y = pd.DataFrame([0 if x == True else 1 for x in list(tiva['class'])], columns=['class'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling
oversample = RandomOverSampler(sampling_strategy='minority', random_state=123)
X_res, y_res = oversample.fit_resample(X_train, y_train)

# Train the XGBoost model
clf = xgb.XGBClassifier(random_state=42)
clf.fit(X_res, y_res.values.ravel())

# Make predictions on the test data
pred = clf.predict(X_test)

# Select data points from the test set where class is 1
check_Data = pd.concat([X_test, y_test], axis=1)
check_Data_1 = check_Data[check_Data['class'] == 1].drop(['class'], axis=1)

# Create SHAP explainer and compute SHAP values
explainer = shap.Explainer(clf)
shap_values = explainer(check_Data_1)

# Generate SHAP waterfall plots
for i in range(len(check_Data_1)):
    shap.plots.waterfall(shap_values[i])
