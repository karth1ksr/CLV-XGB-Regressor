# Customer Lifetime Value Prediction using XGBoost Regression

## Overview

This project builds a predictive model to estimate **Customer Lifetime Value (CLV)** using the XGBoost regression algorithm. CLV is a critical metric in marketing and customer relationship management — it helps businesses identify high-value customers and tailor their strategies accordingly.

---

## Objectives

- Predict CLV based on customer and policy features
- Perform exploratory data analysis (EDA) to uncover patterns
- Evaluate model performance using regression metrics
- Derive actionable insights to improve customer acquisition and retention

---

## Dataset

- **Source**: Auto insurance dataset (FROM [abdurahman-riyad](https://github.com/abdulrahman-riyad/Customer-Lifetime-Value-Regression-Model))
- **Target Variable**: Customer Lifetime Value (`CLV`)
- **Features**:
  - Demographics: Age, Gender, Income
  - Policy details: Number of policies, Coverage type
  - Behavior: Claim history, Vehicle type, Region, etc.

---


## Exploratory Data Analysis (EDA)

Performed detailed EDA to:
- Understand feature distributions
- Detect outliers
- Visualize correlations and feature importance
- Identify business-driven patterns and segmentations (e.g., high CLV clusters)

---

## Machine Learning Pipeline

1. **Data Cleaning**
   - Handled missing values and inconsistent types
   - Converted categorical variables using encoding

2. **Feature Engineering**
   - Normalization, transformation, and correlation checks

3. **Modeling**
   - Applied **XGBoost Regressor**
   - Performed hyperparameter tuning

4. **Evaluation**
   - RMSE, MAE, and R² Score
   - Residual analysis and prediction error checks

---

## Results

| Metric | Score |
|--------|-------|
| RMSE   | 0.2009 |
| R²     | 0.90   |

- The model shows high predictive accuracy, with age, number of policies, and premium per policy emerging as key features.

---

## Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- RandomSearchCV, GridSearchCV
- XGBoost

---

## Folder Structure
<pre><code>

clv-xgboost-regression/
├── data/ # Dataset
├── notebooks/ # Jupyter notebooks with EDA and modeling
├── xgb_clv_model.pkl # Trained XGBoost model ( with all the features )
├── xgb_clv_imp_model.pkl # Trained XGBoost model ( with important features)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

</code></pre>

## Future Improvements

- Deploy model via Flask or FastAPI
- Integrate CLV predictions into CRM dashboard
- Try other regressors (LightGBM, Random Forest) for comparison

---

## License

This project is licensed under the **[MIT License](LICENSE)**.

---

## Acknowledgements

- Inspired by CLV modeling techniques in insurance and fintech sectors
- Thanks to [XGBoost](https://xgboost.readthedocs.io/) for powerful regression tools.
