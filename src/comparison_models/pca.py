import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load the data
tiva = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')

# Separate features (X) and target (y)
X = tiva.drop(['class'], axis=1)
y = pd.DataFrame([0 if x else 1 for x in tiva['class']], columns=['class'])

for i in range(1, 10):
    print("==========================================")
    print("n_components:", i)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply PCA
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Oversampling
    oversample = RandomOverSampler(sampling_strategy='minority', random_state=123)
    X_res, y_res = oversample.fit_resample(X_train_pca, y_train)

    # Train XGBoost model
    model = XGBClassifier(random_state=42)
    model.fit(X_res, y_res)

    # Predict and evaluate
    y_pred = model.predict(X_test_pca)
    F1_score_val = f1_score(y_test.values.ravel(), y_pred, average='macro')
    print("F1_score:", F1_score_val)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["True: MACE", "False: not-MACE"], digits=3))

# Plot the explained variance ratio
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
