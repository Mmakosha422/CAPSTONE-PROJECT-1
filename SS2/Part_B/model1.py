import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. Set the data directory
# ---------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# 2. Load the feature-engineered training and testing data
# ---------------------------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train_engineered.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_engineered.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()


print("Feature-engineered data loaded successfully.")

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ---------------------------------------------------------
# 3. Identify categorical and numerical features
# ---------------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()


print("\nCategorical features:", len(categorical_columns))
print("Numerical features:", len(numerical_columns))


# ---------------------------------------------------------
# 4. Create preprocessing pipelines
# ---------------------------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_columns),
    ("numerical", numerical_pipeline, numerical_columns)
])


# ---------------------------------------------------------
# 5. Create Model 1: Logistic Regression
# ---------------------------------------------------------

logistic_regression = LogisticRegression(
    C=1.0,
    solver="liblinear",
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)


# ---------------------------------------------------------
# 6. Create the complete modelling pipeline
# ---------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", logistic_regression)
])


# ---------------------------------------------------------
# 7. Train the model
# ---------------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ---------------------------------------------------------
# 8. Generate predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# 9. Evaluate the model
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ---------------------------------------------------------
# 10. Display evaluation results
# ---------------------------------------------------------

print("\n==============================")
print("MODEL 1: LOGISTIC REGRESSION")
print("==============================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------------------------
# 11. Display model hyperparameters
# ---------------------------------------------------------

print("\nModel hyperparameters:")
print("Penalty: l2")
print("C: 1.0")
print("Solver: liblinear")
print("Class weight: balanced")
print("Maximum iterations: 1000")
print("Random state: 42")


print("\nModel 1 training and inference completed successfully.")