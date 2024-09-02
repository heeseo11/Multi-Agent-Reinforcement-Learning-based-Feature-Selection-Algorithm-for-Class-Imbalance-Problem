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

## Reward Definition

**1. F1-score**

The F1-score was set as the primary reward to evaluate the overall performance of the XGBoost model, and rewards were adjusted based on the difference in performance between previous and current episodes.


**2. Mutual Information (MI)**
 
MI was used to assign higher rewards to variables that significantly influence the target variable, facilitating faster learning and convergence.
   
**3. Shapley Additive Explanations (SHAP)**

SHAP values were utilized to assign additional rewards to variables with high explanatory power for the minority class, enhancing the model's focus on minority class predictions.
