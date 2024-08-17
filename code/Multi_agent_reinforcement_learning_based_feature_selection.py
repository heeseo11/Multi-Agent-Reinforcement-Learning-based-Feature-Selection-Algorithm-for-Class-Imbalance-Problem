import shap
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load the data
tiva = pd.read_csv('data.csv', index_col=0,  encoding='UTF-8')
mutual_df = pd.read_csv('mutual_info_result.csv', index_col=0)
mutual_df = mutual_df[mutual_df['col_name'] != 'class'].reset_index(drop=True)

# Drop duplicate columns
drop_list = list(set(tiva.columns) - set(tiva.T.drop_duplicates(keep='first').T.columns))
tiva = tiva.drop(drop_list, axis=1)

# Separate features (X) and target (y)
X = tiva.drop(['class'], axis=1)
y = pd.DataFrame([0 if x else 1 for x in tiva['class']], columns=['class'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling
oversample = RandomOverSampler(sampling_strategy='minority', random_state=123)
X_res, y_res = oversample.fit_resample(X_train, y_train)

# Function to compute F1-score, feature importance, and SHAP values
def reward_weight(input_features):
    x_train = X_res.iloc[:, input_features]
    x_test = X_test.iloc[:, input_features]
    
    clf = xgb.XGBClassifier(random_state=42)
    clf.fit(x_train, y_res.values.ravel())
    
    pred = clf.predict(x_test)
    F1_score_val = f1_score(y_test.values.ravel(), pred, average='macro')

    explainer = shap.Explainer(clf)
    shap_values = explainer(x_test)
    shap_result = pd.DataFrame(shap_values.values)
    shap_weight = list(shap_result.abs().mean())

    print("F1_score : ", F1_score_val)

    return F1_score_val, clf.feature_importances_, shap_weight

# Function to get reward based on selected features
def get_reward(features):
    if len(features) == 0:
        return 0
    f1_score_val, importance, shap_weight = reward_weight(features)
    return f1_score_val * 100, importance, shap_weight

# Q-learning parameters
Q_values = [[-1, -1] for _ in range(250)]
epsilon = 0.05
alpha = 0.2
epsilon_decay_rate = 0.995
alpha_decay_rate = 0.995

# Initialization
num_episodes = 1500
num_agents = 250
current_actions = [0] * num_agents
previous_action = [1] * num_agents
previous_R = 81.28

for episode in range(num_episodes):
    print("--------------------------------------------------")
    print("num_episodes : ", episode)

    for agent in range(num_agents):
        rand_number = random.uniform(0, 1)
        current_actions[agent] = np.argmax(Q_values[agent]) if rand_number > epsilon else random.choice([0, 1])

    selected_features = [i for i, act in enumerate(current_actions) if act == 1]
    Current_R, feature_importance, shap_weight = get_reward(selected_features)
    print("R:", Current_R)

    different_indices = [i for i in range(len(current_actions)) if current_actions[i] != previous_action[i]]

    # Improvement in Reward
    Improvement_R = Current_R - previous_R
    divide_R = Improvement_R / len(different_indices) if len(different_indices) > 0 else 0

    mapped_shap_weight = np.zeros(len(current_actions))
    indices = [i for i, x in enumerate(current_actions) if x == 1]

    for i, weight in zip(indices, shap_weight):
        mapped_shap_weight[i] = weight

    total_mapped_shap_weight = sum(mapped_shap_weight)
    mutual_info_values = mutual_df.loc[different_indices, 'mutual_info']
    total_mutual_info_values = sum(mutual_info_values)

    # Update Q-values
    for agent in different_indices:
        if Improvement_R > 0:
            Q_values[agent][current_actions[agent]] += alpha * (
                divide_R + (mapped_shap_weight[agent] / total_mapped_shap_weight) + 
                (mutual_info_values[agent] / total_mutual_info_values) - Q_values[agent][current_actions[agent]]
            )
        else:
            Q_values[agent][current_actions[agent]] += alpha * (divide_R - Q_values[agent][current_actions[agent]])

    # Update for next iteration
    previous_R = Current_R.copy()
    previous_action = current_actions.copy()
    alpha *= alpha_decay_rate
    epsilon *= epsilon_decay_rate

    print("alpha:", alpha)
    print("epsilon:", epsilon)
    print("Q_values:", Q_values)

# Final feature selection and model evaluation
last_feature = [np.argmax(Q_values[i]) for i in range(num_agents)]
selected_features = [X_train.columns[i] for i in range(num_agents) if last_feature[i] == 1]

select_X_res = X_res[selected_features]
clf = xgb.XGBClassifier()
clf.fit(select_X_res, y_res.values.ravel())

select_X_test = X_test[selected_features]
pred = clf.predict(select_X_test)

F1_score_val = f1_score(y_test.values.ravel(), pred, average='macro')
print("F1_score:", F1_score_val)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["True: MACE", "False: not-MACE"], digits=4))
