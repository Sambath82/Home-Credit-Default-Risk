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

![target_dist_sampling](https://github.com/user-attachments/assets/66bbe5e4-e4a1-495e-a251-a4d61cb860a7)

We evaluated our models using ROC AUC, as it provides a reliable measure for binary classification tasks—especially when the labels are imbalanced, with the `default` class being the minority. We tested four models: **HistGradientBoosting (HGBT)**, **LightGBM with SMOTE**, **LightGBM with class weighting**, and **CatBoost with class weighting**.

Among these, **CatBoost with class weighting** stood out as the top performer, achieving the highest AUC-ROC score of around **0.73**. This score highlights CatBoost’s strength in distinguishing between default and non-default cases, making it the best model for this task. After careful tuning, CatBoost reached peak performance with just **147 iterations** and achieved a test AUC-ROC of **0.75**, reflecting its reliability and accuracy in managing class imbalance effectively.

## Model Explainability

![shap_values](https://github.com/user-attachments/assets/6d11763c-444b-42a7-a081-97abbc22b551)

The chosen model demonstrates a balanced and insightful distribution of feature importance, with the top three drivers being `'EXT_SOURCE_3'`, `'EXT_SOURCE_2'`, and `'CREDIT_GOODS_RATIO'`. These features provide intuitive and valuable insights—higher values of these features are likely linked to a lower risk of default. This aligns with expectations based on financial stability indicators. Notably, the engineered features have also shown their impact by ranking 4th, 5th, and 6th in importance, validating the feature engineering efforts as successful. This indicates that our custom features capture meaningful patterns in the data, which strengthen the model’s ability to differentiate between default and non-default cases, especially within an imbalanced dataset. 

By capturing critical insights from both original and engineered features, the model is better positioned to make robust predictions, showing that our approach has truly added value.

## Metrics

![thresholds](https://github.com/user-attachments/assets/f0ddd650-d4ce-4f96-a841-33d1ef523bd3)

To evaluate the business impact of our model, we begin by setting an optimal threshold for our classifier. After analyzing different thresholds, we determined that the highest achievable F1 score is **0.29**. However, for this project, we have decided to prioritize recall over precision, as the business aims to identify as many default cases as possible, even at the cost of some over-prediction. As a result, we have set the threshold at **0.62**, which gives us the following performance metrics:

| Threshold  | 0.62 |
|------------|------|
| Precision  | 0.23 |
| Recall     | 0.38 |
| F1 Score   | 0.29 |
| Alert Rate | 0.13 |

While this choice aligns with the business goal of maximizing recall, it is important to recognize some areas that need attention. The low precision (0.23) and F1 score (0.29) suggest that the model might be over-predicting the default class, leading to more false positives. Additionally, the alert rate of 0.13 means that only 13% of the predictions are true positives, indicating that there is room for improvement.

These results highlight the need for further refinement in the model, possibly through better feature engineering, data quality improvements, or addressing any imbalances in the dataset. By optimizing these factors, we can enhance the model's performance, achieving more accurate predictions while maintaining the business focus on high recall.


### Summary

Predicting default risk for Home Credit is a challenging task, especially when it comes to accurately identifying the minority "default" cases. Despite our efforts in feature engineering to capture patterns linked to defaults, the features showed only weak correlations with the target, limiting their impact. To improve accuracy, we need to create new features with stronger correlations to the target. **Automated feature engineering tools, such as Deep Feature Synthesis (DFS),** could be a valuable next step. In this project, DFS was used with its default settings, which apply primitives across all dataframes and columns. However, this behavior can be fine-tuned through various parameters. For example, specific dataframes and columns can be excluded or included on a per-primitive basis, providing more control over the features and reducing runtime overhead. Additionally, DFS supports advanced custom primitives, allowing you to specify extra arguments to create more complex features. This flexibility could uncover hidden patterns and potentially enhance model performance.

We optimized our model using **CatBoost with class weighting** and set the threshold at **0.62**. The model achieved the following performance metrics:

- **F1 Score**: 0.29
- **Recall**: 0.38
- **Precision**: 0.23
- **Alert Rate**: 0.13
- **AUC-ROC**: 0.7373

These results reflect notable improvements over the baseline, demonstrating the model’s enhanced ability to distinguish between defaults and non-defaults. While precision is still relatively low (0.23), the recall of 0.38 aligns with the business goal of prioritizing the identification of defaults. The alert rate of 0.13 suggests room for improvement in reducing false positives and further refining the model.

The confusion matrix below provides additional insight into the model’s performance:
- **True Positives (TP)**: 283 — Correctly identified defaults
- **True Negatives (TN)**: 7527 — Correctly identified non-defaults
- **False Positives (FP)**: 947 — Non-defaults incorrectly flagged as defaults
- **False Negatives (FN)**: 469 — Defaults that went unflagged

It is important to note that the analysis was based on only a **10% random sample** of the data. Expanding the dataset could provide the model with more information, potentially improving its accuracy. Additionally, the **data preprocessing** steps were not fully optimized. With more thorough preprocessing, we could create better features that may capture stronger correlations with the target, further improving model performance.

Overall, these results demonstrate the model’s potential for predicting default risk, but there is still room for improvement. Techniques like DFS, more refined data preprocessing, and utilizing the full dataset could lead to even more accurate and reliable outcomes.