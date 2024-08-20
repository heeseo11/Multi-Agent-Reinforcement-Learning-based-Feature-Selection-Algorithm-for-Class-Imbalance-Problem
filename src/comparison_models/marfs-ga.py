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
tiva = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')

# Remove duplicate columns
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

# Define accuracy function
def accuracy(input_features):
    x_train = X_res.iloc[:, input_features]
    x_test = X_test.iloc[:, input_features]

    clf = xgb.XGBClassifier(random_state=42)
    clf.fit(x_train, y_res.values.ravel())

    pred = clf.predict(x_test)
    F1_score_val = f1_score(y_test.values.ravel(), pred, average='macro')

    print("F1_score:", F1_score_val)

    return F1_score_val

# Define reward function
def get_reward(features):
    if len(features) == 0:
        return 0
    return accuracy(features) * 100

# Initialize Q-values and parameters
Q_values = [[-1, -1] for _ in range(250)]
epsilon = 0.4
alpha = 0.2
epsilon_decay_rate = 0.995
alpha_decay_rate = 0.995
num_episodes = 500
num_agents = 250

# Q-learning loop
for episode in range(num_episodes):
    print("--------------------------------------------------")
    print("Episode:", episode)

    m_actions = [0] * num_agents
    g_actions = [0] * num_agents

    for agent in range(num_agents):
        rand_number = random.uniform(0, 1)
        if rand_number > epsilon:
            m_actions[agent] = np.argmax(Q_values[agent])
            g_actions[agent] = random.choice([0, 1])
        else:
            m_actions[agent] = random.choice([0, 1])
            g_actions[agent] = random.choice([0, 1])

    total_m_model = [i for i, act in enumerate(m_actions) if act == 1]
    total_g_model = [i for i, act in enumerate(g_actions) if act == 1]

    m_R = get_reward(total_m_model)
    g_R = get_reward(total_g_model)

    R = m_R - g_R
    print("Final Reward:", R)

    for agent in range(num_agents):
        if m_actions[agent] != g_actions[agent]:
            Q_values[agent][m_actions[agent]] += alpha * (R - Q_values[agent][m_actions[agent]])

    alpha *= alpha_decay_rate
    epsilon *= epsilon_decay_rate
    print("Alpha:", alpha)
    print("Epsilon:", epsilon)
    print("Total M Model F1:", m_R)
    print("==========================================")

# Final feature selection and model evaluation
last_feature = [np.argmax(Q_values[i]) for i in range(num_agents)]
selected_features = [X_train.columns[i] for i in range(num_agents) if last_feature[i] == 1]

select_X_res = X_res[selected_features]
select_X_test = X_test[selected_features]

clf = xgb.XGBClassifier(random_state=42)
clf.fit(select_X_res, y_res.values.ravel())

pred = clf.predict(select_X_test)
F1_score_val = f1_score(y_test.values.ravel(), pred, average='macro')
print("F1_score:", F1_score_val)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["True: MACE", "False: not-MACE"], digits=3))
