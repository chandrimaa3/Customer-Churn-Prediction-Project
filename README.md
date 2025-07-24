# Telecom Customer Churn Prediction 

## 1. Project Overview

This project addresses a critical business problem for a telecom company: **customer churn**. By leveraging machine learning, it builds a predictive model to identify customers who are most likely to cancel their subscriptions. The ultimate goal is to enable the company to take proactive steps, such as offering targeted incentives, to retain these at-risk customers and reduce revenue loss.

This repository contains the complete Python script for an end-to-end data science workflow, from data cleaning and exploratory analysis to model training, evaluation, and interpretation.

---

## 2. Dataset

The dataset used for this project is the **Telco Customer Churn** dataset from Kaggle. It contains information about 7,043 customers and includes details on:
* **Customer Demographics:** Gender, senior citizen status, partner, and dependents.
* **Subscribed Services:** Phone, multiple lines, internet, online security, online backup, etc.
* **Account Information:** Customer tenure, contract type, payment method, and monthly/total charges.
* **Target Variable:** The `Churn` column, which indicates whether the customer left within the last month.

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 3. Tech Stack & Libraries

* **Python 3.x**
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Matplotlib & Seaborn:** For data visualization and exploratory data analysis.
* **Scikit-learn:** For data preprocessing, building machine learning models (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`), and model evaluation.

---

## 4. Project Workflow

The project follows a structured data science methodology:

1.  **Data Loading & Cleaning:** The dataset was loaded and inspected. The `TotalCharges` column was converted to a numeric type, and missing values were imputed using the column's **median**. The non-predictive `customerID` was dropped.

2.  **Exploratory Data Analysis (EDA):** Visualizations were created to understand the relationships between different features and customer churn. Key insights revealed that customers on **month-to-month contracts** and those with **shorter tenures** were significantly more likely to churn.

3.  **Feature Engineering & Preprocessing:** A `ColumnTransformer` pipeline was created to systematically handle data preprocessing.
    * **Numerical Features** (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) were scaled using `StandardScaler`.
    * **Categorical Features** (like `Contract`, `InternetService`, etc.) were converted into a numerical format using `OneHotEncoder`.

4.  **Model Building & Training:** The data was split into training (80%) and testing (20%) sets. Several classification models were trained to establish a performance baseline:
    * Logistic Regression
    * Random Forest Classifier
    * Gradient Boosting Classifier

5.  **Model Evaluation:** Models were evaluated on the test set using key metrics like the **Classification Report** (Precision, Recall, F1-score) and the **ROC AUC Score**, which is particularly suitable for imbalanced datasets like churn prediction.

6.  **Hyperparameter Tuning:** Based on the initial results, the **Gradient Boosting Classifier** was selected for optimization. `GridSearchCV` was used to systematically search for the best combination of hyperparameters (`n_estimators`, `learning_rate`, `max_depth`), further improving the model's predictive power.

---

## 5. Key Results

* The final **Tuned Gradient Boosting Classifier** emerged as the best model with a **ROC AUC score of approximately 0.846** on the unseen test data.
* **Feature Importance Analysis** from the final model revealed the top predictors of churn:
    1.  **Contract Type:** `Contract_Month-to-month` was the single strongest indicator.
    2.  **Tenure:** Shorter customer tenure is a high-risk factor.
    3.  **Internet Service:** `InternetService_Fiber optic` was a significant predictor.
    4.  **Total Charges:** The total amount charged to the customer.

---

## 6. How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    Create a `requirements.txt` file with the following content:
    ```
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    ```
    Then run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Python script or Jupyter Notebook:**
    ```bash
    python your_script_name.py
    ```
    or open and run the `Customer Churn Prediction Project.ipynb` file in a Jupyter environment. The script will execute all the steps and print the evaluation results and visualizations.

---

## 7. Conclusion & Business Insights

The predictive model developed in this project can be a valuable asset for the telecom company. By integrating this model, the business can:

* **Proactively Identify At-Risk Customers:** Generate a regular list of customers with a high probability of churning.
* **Implement Targeted Retention Strategies:** The marketing team can use this list to offer personalized discounts, plan upgrades, or other incentives to encourage customers to stay.
* **Optimize Marketing Spend:** Focus retention efforts on customers who are most likely to leave, leading to a higher return on investment for marketing campaigns.

The key business recommendation is to focus retention efforts on customers with **short tenures** who are on **month-to-month contracts**, as they represent the highest risk segment.
