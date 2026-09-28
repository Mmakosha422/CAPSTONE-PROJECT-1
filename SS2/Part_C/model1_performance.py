import pandas as pd
import numpy as np
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

PART_B_DIR = Path(__file__).resolve().parent.parent / "Part_B"


# ---------------------------------------------------------
# 2. Load the feature-engineered data
# ---------------------------------------------------------

X_train = pd.read_csv(
    PART_B_DIR / "X_train_engineered.csv"
)

X_test = pd.read_csv(
    PART_B_DIR / "X_test_engineered.csv"
)

y_train = pd.read_csv(
    PART_B_DIR / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    PART_B_DIR / "y_test.csv"
).squeeze()


print("Model 1 performance data loaded successfully.")

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ---------------------------------------------------------
# 3. Identify categorical and numerical variables
# ---------------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()


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
# 5. Create the Logistic Regression model
# ---------------------------------------------------------

logistic_regression = LogisticRegression(
    C=1.0,
    solver="liblinear",
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", logistic_regression)
])


# ---------------------------------------------------------
# 6. Train Model 1
# ---------------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ---------------------------------------------------------
# 7. Generate predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# 8. Calculate performance metrics
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
# 9. Bootstrap confidence intervals
# ---------------------------------------------------------

def bootstrap_metric(
    y_true,
    y_pred,
    y_prob,
    metric_name,
    n_bootstrap=1000,
    random_state=42
):
    """
    Calculate a bootstrap 95% confidence interval
    for a classification performance metric.
    """

    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    scores = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            len(y_true),
            len(y_true)
        )

        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        sample_prob = y_prob[indices]

        # ROC-AUC and PR-AUC require both target classes
        if metric_name in ["roc_auc", "pr_auc"]:
            if len(np.unique(sample_true)) < 2:
                continue

        if metric_name == "accuracy":
            score = accuracy_score(
                sample_true,
                sample_pred
            )

        elif metric_name == "precision":
            score = precision_score(
                sample_true,
                sample_pred,
                zero_division=0
            )

        elif metric_name == "recall":
            score = recall_score(
                sample_true,
                sample_pred,
                zero_division=0
            )

        elif metric_name == "f1":
            score = f1_score(
                sample_true,
                sample_pred,
                zero_division=0
            )

        elif metric_name == "roc_auc":
            score = roc_auc_score(
                sample_true,
                sample_prob
            )

        elif metric_name == "pr_auc":
            score = average_precision_score(
                sample_true,
                sample_prob
            )

        scores.append(score)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return lower, upper


# ---------------------------------------------------------
# 10. Calculate 95% confidence intervals
# ---------------------------------------------------------

metrics = {
    "Accuracy": (
        accuracy,
        "accuracy"
    ),
    "Precision": (
        precision,
        "precision"
    ),
    "Recall": (
        recall,
        "recall"
    ),
    "F1-score": (
        f1,
        "f1"
    ),
    "ROC-AUC": (
        roc_auc,
        "roc_auc"
    ),
    "PR-AUC": (
        pr_auc,
        "pr_auc"
    )
}


print("\n======================================")
print("MODEL 1 PERFORMANCE")
print("LOGISTIC REGRESSION")
print("======================================")

print("\nPerformance metrics with 95% bootstrap confidence intervals:")

for name, (value, metric_name) in metrics.items():

    lower, upper = bootstrap_metric(
        y_test,
        y_pred,
        y_prob,
        metric_name,
        n_bootstrap=1000,
        random_state=42
    )

    print(
        f"{name}: {value:.4f} "
        f"(95% CI: {lower:.4f} - {upper:.4f})"
    )


# ---------------------------------------------------------
# 11. Display confusion matrix
# ---------------------------------------------------------

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------------------------
# 12. Display sample size
# ---------------------------------------------------------

print("\nTest set size:", len(y_test))

print("\nModel 1 performance calculations completed successfully.")