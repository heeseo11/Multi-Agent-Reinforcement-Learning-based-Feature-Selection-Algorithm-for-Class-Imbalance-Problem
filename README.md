# Feature Selection integrating Shapley Values and Mutual Information in Reinforcement Learning: An Application to Prediction of Post-operative Outcomes in End-stage Renal Disease Patients

[paper link 넣기]

The overall structure of the multi-agent reinforcement learning framework is as follows:

- Multi-agent reinforcement learning

<img src="https://github.com/user-attachments/assets/d1653187-c857-403d-a9df-b7eb07b41285" width="600">

<p>Fig1. MARL process selecting actions by multi-agents and apportioning rewards for these actions, utilizing SHAP and MI for reward distribution.</p>

---

## Agent Definition
In this study, each of the 250 agents corresponds to one feature in the dataset, and their role is to select the optimal feature combination using a q-value to evaluate the effectiveness of their actions.

## Action Definition
An action determines whether a specific variable is selected (1) or not (0) for feature selection, with the choice impacting subsequent states and rewards in the reinforcement learning process.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/a962eaea-2bcb-42ef-8e1e-3fb07aaf094e">

## Reward Definition

**1. F1-score**

The F1-score was set as the primary reward to evaluate the overall performance of the XGBoost model, and rewards were adjusted based on the difference in performance between previous and current episodes.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/9876d417-5b75-48cd-a6a1-280bdf4daaad">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/75d59de1-8b1e-4438-8ca0-5b59d9e97f46">

**2. Mutual Information (MI)**
 
MI was used to assign higher rewards to variables that significantly influence the target variable, facilitating faster learning and convergence.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/52b0dc0d-0dd6-4696-bc3b-311c43fb4610">

   
**3. Shapley Additive Explanations (SHAP)**

SHAP values were utilized to assign additional rewards to variables with high explanatory power for the minority class, enhancing the model's focus on minority class predictions.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/85057311-9987-496a-812b-54424452d157">
