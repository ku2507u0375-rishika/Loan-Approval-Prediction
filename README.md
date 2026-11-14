# Loan Approval Prediction Using Decision Tree Classifier

## 1. Project Overview

The Loan Approval Prediction project is a Machine Learning-based classification system designed to predict whether a loan application is likely to be approved or rejected based on applicant information.

Banks and financial institutions consider several factors before approving a loan, including income, employment experience, credit score, loan amount, loan purpose, and previous loan defaults.

This project uses a **Decision Tree Classifier** to learn patterns from the Loan Approval Classification Dataset and predict the loan status of an applicant.

A web application is also developed using Streamlit, allowing users to enter applicant details and obtain a prediction through an interactive interface.

## 2. Problem Statement

Banks and financial institutions receive numerous loan applications. Evaluating applications requires examining different personal, financial, and credit-related factors.

The objective of this project is to develop a Machine Learning classification model that predicts whether a loan application is likely to be approved or rejected based on the applicant's details.

## 3. Project Objectives

- To understand and apply supervised Machine Learning concepts.
- To explore and analyze the loan approval dataset.
- To perform data cleaning and preprocessing.
- To visualize patterns and relationships in the dataset.
- To train a Decision Tree Classifier.
- To evaluate the model using classification performance metrics.
- To identify important features influencing model predictions.
- To develop an interactive Streamlit application.
- To deploy the application online.

## 4. Dataset Description

**Dataset Name:** Loan Approval Classification Dataset

**Source:** Kaggle

**Dataset Link:** https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

The original dataset contains 45,000 records and 14 columns.

After data cleaning, the dataset contains 44,990 records and 14 columns.

### Input Features

The model uses 13 input features:

| Feature | Description |
|---|---|
| person_age | Age of the applicant |
| person_gender | Gender of the applicant |
| person_education | Education level |
| person_income | Annual income |
| person_emp_exp | Employment experience |
| person_home_ownership | Home ownership status |
| loan_amnt | Requested loan amount |
| loan_intent | Purpose of the loan |
| loan_int_rate | Loan interest rate |
| loan_percent_income | Loan amount as a proportion of income |
| cb_person_cred_hist_length | Credit history length |
| credit_score | Applicant's credit score |
| previous_loan_defaults_on_file | Previous loan default history |

**Target Variable:** `loan_status`

The target variable has two classes:

- 0 – Rejected
- 1 – Approved

## 5. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Model development |
| Pandas | Data handling and preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-learn | Model training and evaluation |
| Joblib | Saving and loading the trained model |
| Streamlit | Web application development |
| GitHub | Project repository |
| Streamlit Community Cloud | Application deployment |

## 6. Data Exploration and Visualization

Exploratory Data Analysis (EDA) was performed to understand the dataset and identify patterns in applicant information.

The notebook includes the following visualizations:

1. Loan Approval Distribution
2. Income Distribution
3. Credit Score Distribution
4. Loan Amount vs Loan Status
5. Outlier Detection in Person Income
6. Correlation Heatmap
7. Employment Experience vs Loan Status
8. Loan Approval by Loan Intent
9. Credit Score vs Loan Amount

These visualizations help examine the relationship between applicant information and loan status.

## 7. Data Cleaning and Preprocessing

The following preprocessing operations were performed:

### 7.1 Removing Duplicate Records

Duplicate rows were removed from the dataset.

### 7.2 Filtering Invalid Values

Records with applicant age greater than 100 years or employment experience greater than 60 years were removed.

### 7.3 Target Variable Cleaning

The target column was converted into numeric format. Rows with missing or invalid target values were removed, retaining only classes 0 and 1.

### 7.4 Handling Missing Values

- Numerical missing values were handled using the median.
- Categorical missing values were handled using the mode.
- The median and mode were calculated using the training dataset.

### 7.5 Feature and Target Separation

The input features were stored in `X`, while the target variable `loan_status` was stored in `y`.

### 7.6 Train-Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

The split used `random_state=42` and stratification.

### 7.7 Categorical Encoding

Categorical features were converted into numerical columns using One-Hot Encoding with `drop_first=True`.

After encoding, the training and testing datasets contained 22 features.

## 8. Machine Learning Algorithm

### Decision Tree Classifier

A Decision Tree Classifier is a supervised Machine Learning algorithm used for classification.

It works by dividing data into smaller groups based on feature values. Each internal node represents a decision, branches represent possible outcomes, and leaf nodes represent the predicted class.

In this project, the algorithm predicts one of two loan application statuses:

- Approved
- Rejected

### Model Configuration

The following parameters were used:

| Parameter | Value |
|---|---|
| Algorithm | Decision Tree Classifier |
| max_depth | 8 |
| random_state | 42 |
| class_weight | balanced |

The model was trained using the preprocessed training dataset.

## 9. Model Evaluation

The trained model was evaluated using the testing dataset.

The following evaluation metrics were obtained:

| Evaluation Metric | Result |
|---|---:|
| Training Accuracy | 89.07% |
| Testing Accuracy | 88.28% |
| Precision | 67.84% |
| Recall | 89.85% |
| F1-Score | 77.31% |

### Evaluation Metrics Explained

**Accuracy:** Measures the overall percentage of correct predictions.

**Precision:** Measures how many applications predicted as approved were actually approved.

**Recall:** Measures how many actual approved applications were correctly identified.

**F1-Score:** Represents the balance between precision and recall.

### Confusion Matrix

A confusion matrix was generated to compare actual loan statuses with predicted loan statuses.

The notebook also includes a classification report showing precision, recall, F1-score, and support for both classes.

## 10. Feature Importance Analysis

Feature importance was calculated using the trained Decision Tree model.

The most important features identified were:

| Feature | Importance |
|---|---:|
| Previous Loan Defaults | 0.6716 |
| Loan Interest Rate | 0.1227 |
| Loan Percent Income | 0.0819 |
| Person Income | 0.0687 |

The results show that previous loan default history has the highest feature importance in this trained model.

A bar chart of the top 10 important features and a visualization of the Decision Tree are also included in the notebook.

## 11. Streamlit Web Application

A web application was developed using Streamlit to provide an interactive interface for the trained Decision Tree model.

### Application Features

- Applicant age input.
- Gender selection.
- Education selection.
- Annual income input.
- Employment experience input.
- Home ownership selection.
- Loan amount input.
- Loan intent selection.
- Loan interest rate input.
- Loan amount-to-income ratio input.
- Credit history length input.
- Credit score input.
- Previous loan defaults selection.
- Loan approval prediction.

### Application Working Process

1. The user opens the Streamlit application.
2. The user enters applicant information.
3. The entered details are converted into a Pandas DataFrame.
4. Missing values are handled using saved training median and mode values.
5. Categorical data is converted using One-Hot Encoding.
6. The encoded columns are aligned with the model's training features.
7. The trained Decision Tree model predicts the loan status.
8. The application displays APPROVED or REJECTED.

The application uses the saved model and preprocessing information stored in `model_bundle.joblib`.

## 12. Project Workflow

```text
Dataset Collection
        |
        v
Data Loading and Exploration
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Feature and Target Separation
        |
        v
Train-Test Split
        |
        v
Missing Value Handling
        |
        v
One-Hot Encoding
        |
        v
Decision Tree Model Training
        |
        v
Model Evaluation
        |
        v
Feature Importance Analysis
        |
        v
Model Saving
        |
        v
Streamlit Web Application
        |
        v
Online Deployment
        |
        v
Loan Approval Prediction
```

## 13. Project Structure

```text
Loan-Approval-Prediction/
│
├── app.py
├── model_bundle.joblib
├── requirements.txt
└── README.md
```

| File | Description |
|---|---|
| app.py | Streamlit application code |
| model_bundle.joblib | Trained Decision Tree model and saved preprocessing information |
| requirements.txt | Required Python libraries |
| README.md | Project documentation |

## 14. Installation and Execution

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/ku2507u0375-rishika/Loan-Approval-Prediction.git
```

### Step 2: Open the Project Folder

```bash
cd Loan-Approval-Prediction
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the default web browser.

## 15. Live Application and GitHub Repository

**Live Application:**

https://ku2507u0375-rishika-loan-approval-prediction-app-abw1qb.streamlit.app/

**GitHub Repository:**

https://github.com/ku2507u0375-rishika/Loan-Approval-Prediction

The application is deployed using Streamlit Community Cloud.

## 16. Advantages

- Provides quick predictions based on applicant information.
- Demonstrates the practical application of supervised Machine Learning.
- Uses personal, financial, and credit-related features.
- Provides a simple and interactive web interface.
- Makes the application accessible through a web browser.

## 17. Limitations

- Predictions depend on the dataset used for training.
- The model may not perform equally well for every applicant group.
- The prediction does not represent an actual bank's loan approval decision.
- The system is an educational prototype and does not replace professional financial evaluation.

## 18. Future Scope

Possible future improvements include:

- Comparing the Decision Tree model with other classification algorithms.
- Performing hyperparameter tuning.
- Evaluating the model using additional datasets.
- Improving the application interface.
- Adding more detailed prediction analysis.

These are possible future improvements and are not part of the current implementation.

## 19. Conclusion

The Loan Approval Prediction project demonstrates the application of supervised Machine Learning to a financial classification problem.

A Decision Tree Classifier was trained using applicant and loan-related information after performing data cleaning, exploratory data analysis, and preprocessing.

The trained model achieved 88.28% testing accuracy. Model performance was evaluated using accuracy, precision, recall, F1-score, a confusion matrix, and a classification report.

A Streamlit web application was developed and deployed online, allowing users to enter applicant details and receive loan approval predictions.

This project provides practical experience in data analysis, preprocessing, Machine Learning model development, model evaluation, and web application deployment.

## 20. Project Team

**Developed By:**

- Rishika Bambhaniya
- Priyanshi Raval

**Faculty Guide:**

Anushree Paul

## 21. References

1. Kaggle – Loan Approval Classification Dataset: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
2. Pandas Documentation: https://pandas.pydata.org/docs/
3. NumPy Documentation: https://numpy.org/doc/
4. Scikit-learn Documentation: https://scikit-learn.org/
5. Matplotlib Documentation: https://matplotlib.org/
6. Seaborn Documentation: https://seaborn.pydata.org/
7. Streamlit Documentation: https://docs.streamlit.io/

   ---

## 22. Application Screenshots

### Home Page and Applicant Information

![Loan Approval App Homepage](Screenshot%20(666).png)

### Loan Application Input Form

![Loan Application Form](Screenshot%20(667).png)

### Prediction Result

![Loan Approval Prediction Result](Screenshot%20(668).png)
