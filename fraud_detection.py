import pandas as pd
df = pd.read_csv("bank_transaction_fraud_10000.csv")


print("Dataset loaded successfully")
print("Rows:",df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:",
df.duplicated().sum())

print("\nFraud distribution")
print(df["is_fraud"].value_counts())

print("\nFraud percentage:")
print(df["is_fraud"].value_counts(normalize = True)* 100)

print(df.info())

print(df.isnull(). sum())

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (6,4))

sns.countplot(data = df, x = "is_fraud")
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Fraud Status")
plt.ylabel("Number of Transactions")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data = df,
              x = "transaction_type",
              hue = "is_fraud")
plt.title("Fraud by transaction Type")
plt.xticks(rotation = 30)
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(data = df,
             x="merchant_category",
             hue = "is_fraud")
plt.title ("Fraud by Merchant Category")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (10,5))
sns.countplot(data = df,
              x="location",
              hue = "is_fraud")

plt.title("Fraud by Location")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data = df,
              x="payment_method",
              hue = "is_fraud")

plt.title("Fraud by Payment Method")
plt.xticks(rotation = 30)
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(data = df,
              x = "device_type",
              hue = "is_fraud")
plt.title("Fraud by Device Type")
plt.show()

plt.figure(figsize = (8,5))
sns.histplot(
    data = df,
    x="transaction_amount",
    bins = 50
)
plt.title("Transaction Amount Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(
    data = df,
    x="is_fraud",
    y="transaction_amount"

)
plt.title("Transaction Amount: Fraud vs Non-Fraud")
plt.show()

numeric_df = df.select_dtypes(include = np.number)
plt.figure(figsize =(12,8))
sns.heatmap(numeric_df.corr(),
            annot = True,
            cmap ="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()

#feature engineering
df["amount_difference"] = (df["transaction_amount"] - df["previous_transaction_amount"])

df["amount_ratio"] = (df["transaction_amount"] / (df["previous_transaction_amount"] + 1))

df["high_value_transaction"] = (df["transaction_amount"]>=10000).astype(int)

df["high_frequency"] = (df["transactions_last_24h"] >= 7).astype(int)

df["unusual_amount"] = (df["amount_ratio"] > 3).astype(int)

df["transaction_date"] = pd.to_datetime(df["transaction_date"])

df["transaction_month"] = (df["transaction_date"].dt.month)

df["transaction_day"] = (df["transaction_date"].dt.day)

df["transaction_dayofweek"] = (df["transaction_date"].dt.dayofweek)

#Machine Learning
X = df.drop(
    columns = ["is_fraud",
               "transaction_id",
               "customer_id",
               "transaction_date",
               "transaction_time"]
)
y = df["is_fraud"]

#train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size  =  0.20,
    random_state = 42,
    stratify = y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

categorical_columns = [
    "transaction_type",
    "merchant_category",
    "location",
    "device_type",
    "payment_method"
]

numerical_columns  =  [
    "transaction_amount",
    "account_age_days",
    "customer_age",
    "previous_transaction_amount",
    "transactions_last_24h",
    "is_international",
    "amount_difference",
    "amount_ratio",
    "high_value_transaction",
    "high_frequency",
    "unusual_amount",
    "transaction_month",
    "transaction_day",
    "transaction_dayofweek"]

preprocessor = ColumnTransformer(
        transformers =[(
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown = "ignore"
            ),
            categorical_columns
        )]
)

#Logistic Regression
from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
                max_depth=8
            )
        )
    ]
)

decision_tree.fit(X_train, y_train)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest.fit(X_train, y_train)

#XGboost
from xgboost import XGBClassifier

xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
        )
    ]
)

xgb_model.fit(X_train, y_train)

#Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("Accuracy:",
          accuracy_score(y_test, predictions))

    print("Precision:",
          precision_score(y_test, predictions))

    print("Recall:",
          recall_score(y_test, predictions))

    print("F1 Score:",
          f1_score(y_test, predictions))

    print("ROC-AUC:",
          roc_auc_score(y_test, probabilities))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))

evaluate_model(
    logistic_model,
    X_test,
    y_test
)

evaluate_model(
    decision_tree,
    X_test,
    y_test
)

evaluate_model(
    random_forest,
    X_test,
    y_test
)

evaluate_model(
    xgb_model,
    X_test,
    y_test
)    

#Comparing Models
results = []

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "XGBoost": xgb_model
}

for name, model in models.items():

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, prob)
    })

results_df = pd.DataFrame(results)

print(results_df)

print(
    results_df.sort_values(
        "F1",
        ascending=False
    )
)

#ConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay

best_model = xgb_model

ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.title("Fraud Detection Confusion Matrix")
plt.show()


#ROC Curve Display
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.title("ROC Curve")
plt.show()

from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.title("Precision-Recall Curve")
plt.show()

class_weight="balanced"

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import RandomizedSearchCV

import shap
from sklearn.model_selection import RandomizedSearchCV
predictions_df = X_test.copy()
fraud_probability = best_model.predict_proba(X_test)[:, 1]

predictions_df["actual_fraud"] = y_test.values

predictions_df["fraud_probability"] = fraud_probability

predictions_df["predicted_fraud"] = (
    fraud_probability >= 0.5
).astype(int)


predictions_df = X_test.copy()

predictions_df["actual_fraud"] = y_test.values

predictions_df["fraud_probability"] = fraud_probability

predictions_df["predicted_fraud"] = (
    fraud_probability >= 0.5
).astype(int)


prediction_output = df.loc[
    X_test.index
].copy()

prediction_output["fraud_probability"] = fraud_probability

prediction_output["predicted_fraud"] = (
    fraud_probability >= 0.5
).astype(int)

prediction_output.to_csv(
    "fraud_predictions.csv",
    index=False
)

prediction_output = df.loc[
    X_test.index
].copy()

prediction_output["fraud_probability"] = fraud_probability

prediction_output["predicted_fraud"] = (
    fraud_probability >= 0.5
).astype(int)

prediction_output.to_csv(
    "fraud_predictions.csv",
    index=False
)