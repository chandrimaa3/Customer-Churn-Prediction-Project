# Customer Churn Prediction for a Telecom Company 

## Project Overview

This project aims to solve a critical business problem for a telecom company: **customer churn**. By leveraging machine learning, we build a predictive model to identify customers who are most likely to cancel their subscriptions. The ultimate goal is to enable the company to take proactive steps, such as offering targeted incentives, to retain these at-risk customers and reduce revenue loss.

This repository contains the complete Python script for the end-to-end data science workflow, from data cleaning and exploratory analysis to model training, evaluation, and interpretation.

---

## Dataset

The dataset used for this project is the **Telco Customer Churn** dataset, which is publicly available on Kaggle. It contains information about 7,043 customers and includes details on:
* **Customer Demographics:** Gender, age range, and whether they have partners and dependents.
* **Subscribed Services:** Phone, multiple lines, internet, online security, online backup, etc.
* **Account Information:** Customer tenure, contract type, payment method, and monthly/total charges.
* **Target Variable:** The `Churn` column, which indicates whether the customer left within the last month.

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Tech Stack & Libraries

* **Python 3.x**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization and exploratory data analysis.
* **Scikit-learn:** For data preprocessing, model building, and evaluation.

---

## Project Workflow

The project follows a structured data science methodology:

1.  **Data Loading & Cleaning:** The dataset was loaded and inspected. The `TotalCharges` column was converted to a numeric type, and missing values were imputed using the median. The non-predictive `customerID` was dropped.

2.  **Exploratory Data Analysis (EDA):** Visualizations were created to understand the relationships between different features and customer churn. Key insights revealed that customers on **month-to-month contracts** and those with **shorter tenures** were significantly more likely to churn.

3.  **Feature Engineering & Preprocessing:** A `ColumnTransformer` pipeline was created to systematically handle data preprocessing.
    * **Numerical Features:** Scaled using `StandardScaler`.
    * **Categorical Features:** Converted into a numerical format using `OneHotEncoder`.

4.  **Model Building & Training:** Several classification models were trained to establish a baseline performance:
    * Logistic Regression
    * Random Forest
    * Gradient Boosting Classifier

5.  **Model Evaluation:** Models were evaluated based on their performance on the test set. Key metrics included **Classification Report** (Precision, Recall, F1-score) and **ROC AUC Score**, which is suitable for imbalanced datasets.

6.  **Hyperparameter Tuning:** The best-performing baseline model (Gradient Boosting) was selected for optimization. `GridSearchCV` was used to search for the best combination of hyperparameters, further improving the model's predictive power.

---

## Key Results

* The final **Tuned Gradient Boosting Classifier** emerged as the best model with a **ROC AUC score of approximately 0.84** on the test set.
* **Feature Importance Analysis** revealed the top predictors of churn:
    1.  **Contract Type:** Month-to-month contracts are the strongest indicator.
    2.  **Tenure:** Customers with shorter tenure are at a higher risk.
    3.  **Total Charges & Monthly Charges:** Higher charges contribute to churn risk.

---

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is provided.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: To generate this file yourself, you can run `pip freeze > requirements.txt` after installing the libraries manually.)*

4.  **Run the Python script:**
    ```bash
    python churn_prediction.py
    ```
    The script will execute all the steps and print the results and visualizations.

---

## Conclusion & Business Insights

The predictive model developed in this project can be a valuable asset for the telecom company. By integrating this model into their systems, the company can:

* **Proactively Identify At-Risk Customers:** Generate a daily or weekly list of customers with a high probability of churning.
* **Implement Targeted Retention Strategies:** The marketing team can use this list to offer personalized discounts, plan upgrades, or other incentives to encourage customers to stay.
* **Optimize Marketing Spend:** Focus retention efforts on customers who are most likely to leave, leading to a higher return on investment for marketing campaigns.

The key takeaway is that focusing on customers with **short tenures** and on **month-to-month contracts** is crucial for reducing overall churn.
