# Bank Fraud Transaction Detection

## 📌 Project Overview

This project focuses on analyzing bank transactions and identifying potentially fraudulent transactions using data analytics and machine learning.

The project follows an end-to-end workflow using **Excel, MySQL, Python, Machine Learning, and Power BI**.

## 🎯 Business Problem

Financial fraud can cause significant losses for banks and customers. The goal of this project is to analyze transaction data, identify patterns associated with fraudulent transactions, and build a machine learning model that can help detect potentially fraudulent transactions.

## 🛠️ Tools & Technologies

* **Excel** – Data cleaning and preprocessing
* **MySQL** – Data storage and SQL analysis
* **Python** – Data analysis and machine learning
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning
* **Power BI** – Interactive visualization and dashboard
* **Random Forest / Classification Model** – Fraud detection

## 🔄 Project Workflow

```text
Raw Transaction Dataset
        ↓
Excel Data Cleaning
        ↓
MySQL Database
        ↓
SQL Analysis
        ↓
Python Data Analysis
        ↓
Machine Learning Model
        ↓
Power BI Dashboard
        ↓
Fraud Detection Insights
```

## 📊 Dataset

The dataset contains bank transaction-related information used to identify fraudulent transactions.

The target variable is **`is fraud`**:

* `0` – Not Fraud
* `1` – Fraud

The dataset was prepared and analyzed to understand transaction patterns and identify characteristics associated with fraudulent activity.

## 🧹 Data Cleaning

Excel and Python were used for data preparation and cleaning.

The process included:

* Checking missing values
* Checking duplicate records
* Validating data types
* Identifying inconsistent values
* Preparing data for SQL analysis
* Preparing features for machine learning

## 🗄️ SQL Analysis

MySQL was used to store and analyze the transaction dataset.

The SQL analysis included:

* Transaction analysis
* Fraud vs. non-fraud transaction analysis
* Aggregation of transaction data
* Identifying transaction patterns
* Analyzing fraud across different categories
* Filtering and grouping transaction records

## 🤖 Machine Learning

A classification-based machine learning approach was used to identify potentially fraudulent transactions.

The model was evaluated using multiple classification metrics because fraud detection is typically an imbalanced classification problem.

Evaluation metrics included:

* Accuracy
* Precision
* Recall
* F1-score

The model achieved approximately **95.75% accuracy** on the evaluated dataset.

However, accuracy alone was not used to evaluate the model because correctly identifying fraudulent transactions is particularly important in an imbalanced dataset.

## 📈 Power BI Dashboard

Power BI was used to create an interactive dashboard for analyzing fraudulent and non-fraudulent transactions.

The dashboard provides insights into:

* Total transactions
* Fraudulent transactions
* Non-fraudulent transactions
* Fraud percentage
* Transaction trends
* Fraud patterns
* Transaction categories
* Other important transaction-level insights

## 💡 Business Use Case

The project can help financial institutions analyze transaction patterns and identify potentially fraudulent transactions.

Possible business applications include:

* Detecting suspicious transactions
* Identifying fraud patterns
* Supporting fraud investigation
* Monitoring transaction activity
* Reducing potential financial losses

## 📁 Project Structure

```text
bank-fraud-transaction-detection/
│
├── README.md
├── data/
├── excel/
├── sql/
├── python/
├── powerbi/
└── images/
```

## 📌 Key Result

The machine learning model achieved approximately **95.75% accuracy** on the evaluated dataset.

Precision, recall, and F1-score were also considered to provide a more meaningful evaluation of fraud detection performance.

## 👨‍💻 Author

**Sarvesh Dudhe**

Electronics Engineering | Data Analyst | Machine Learning

Skills: Python | SQL | Excel | Power BI | Machine Learning

