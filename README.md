# Home Credit Default Risk
![home_credit_default_risk](https://github.com/user-attachments/assets/75520dbb-fcc9-4a57-bbee-c81435a5b5c0)

## Dataset

The dataset for this project is available on [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data), 
provided by Home Credit, a company dedicated to offering loans to the unbanked population. 
It contains a variety of valuable data sources, including loan application details, past credit histories from other institutions, and monthly 
records of previous loans, credit card balances, and payments. These features give us key insights into a client’s financial behavior, 
helping us predict their ability to repay a loan. By leveraging this rich data, we can build a model that not only improves loan repayment 
predictions but also supports greater financial inclusion for underserved individuals.

## Objectives

The main objective of this project is:

> **To develop better credit risk models using a variety of data sources to accurately assess the repayment ability of clients**

To achieve this objective, we’ve broken it down into the following five technical sub-objectives:

1. Prepare and perform a comprehensive exploratory data analysis (EDA) on all datasets, using both tabular and visual techniques.
2. Engineer new predictive features using the automated Featuretools package, and reduce large-dimensional features using the FAMD method.
3. Develop a supervised model to classify behavior into "default" and "non-default" categories.
4. Recommend a threshold that outperforms the current baseline in terms of F1 score and AUC-ROC score.
5. Create an API endpoint for the trained model and deploy it for real-world use.
 
## Model Selection

We evaluated our models using ROC AUC, as it provides a reliable measure for binary classification tasks—especially when the labels are imbalanced, with the `default` class being the minority. We tested four models: **HistGradientBoosting (HGBT)**, **LightGBM with SMOTE**, **LightGBM with class weighting**, and **CatBoost with class weighting**.

Among these, **CatBoost with class weighting** stood out as the top performer, achieving the highest AUC-ROC score of **0.7266**. This score highlights CatBoost’s strength in distinguishing between default and non-default cases, making it the best model for this task. After careful tuning, CatBoost reached peak performance with just **104 iterations** and achieved a test AUC-ROC of **0.73**, reflecting its reliability and accuracy in managing class imbalance effectively.

## Model Explainability

![shap_values](https://github.com/user-attachments/assets/c471d29a-416e-4187-9b5b-ac044e51bb4c)

