import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

DATA_DIR = Path(__file__).resolve().parent.parent / "Part_B"

# Load the same data used for Model 1 and Model 2
X_train = pd.read_csv(DATA_DIR / "X_train_engineered.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_engineered.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("Comparison data loaded successfully.")
print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Identify categorical and numerical columns
categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()


# =========================
# MODEL 1: LOGISTIC REGRESSION
# =========================

categorical_pipeline_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_pipeline_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor_lr = ColumnTransformer([
    ("categorical", categorical_pipeline_lr, categorical_columns),
    ("numerical", numerical_pipeline_lr, numerical_columns)
])

model1 = Pipeline([
    ("preprocessor", preprocessor_lr),
    ("classifier", LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])


# =========================
# MODEL 2: RANDOM FOREST
# =========================

categorical_pipeline_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_pipeline_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor_rf = ColumnTransformer([
    ("categorical", categorical_pipeline_rf, categorical_columns),
    ("numerical", numerical_pipeline_rf, numerical_columns)
])

model2 = Pipeline([
    ("preprocessor", preprocessor_rf),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])


# =========================
# TRAIN MODELS
# =========================

print("\nTraining Model 1...")
model1.fit(X_train, y_train)

print("Training Model 2...")
model2.fit(X_train, y_train)

print("Both models trained successfully.")


# =========================
# PREDICTIONS
# =========================

y_pred_1 = model1.predict(X_test)
y_prob_1 = model1.predict_proba(X_test)[:, 1]

y_pred_2 = model2.predict(X_test)
y_prob_2 = model2.predict_proba(X_test)[:, 1]


# =========================
# CALCULATE METRICS
# =========================

def calculate_metrics(y_true, y_pred, y_prob):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_true, y_prob
        ),
        "PR-AUC": average_precision_score(
            y_true, y_prob
        )
    }


metrics_1 = calculate_metrics(
    y_test,
    y_pred_1,
    y_prob_1
)

metrics_2 = calculate_metrics(
    y_test,
    y_pred_2,
    y_prob_2
)


# =========================
# PAIRED BOOTSTRAP
# =========================

def bootstrap_metric_difference(
    y_true,
    pred1,
    prob1,
    pred2,
    prob2,
    metric_name,
    n_bootstrap=1000,
    random_state=42
):

    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    prob1 = np.asarray(prob1)
    pred2 = np.asarray(pred2)
    prob2 = np.asarray(prob2)

    differences = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            len(y_true),
            len(y_true)
        )

        y_sample = y_true[indices]

        if metric_name == "Accuracy":
            score1 = accuracy_score(
                y_sample,
                pred1[indices]
            )
            score2 = accuracy_score(
                y_sample,
                pred2[indices]
            )

        elif metric_name == "Precision":
            score1 = precision_score(
                y_sample,
                pred1[indices],
                zero_division=0
            )
            score2 = precision_score(
                y_sample,
                pred2[indices],
                zero_division=0
            )

        elif metric_name == "Recall":
            score1 = recall_score(
                y_sample,
                pred1[indices],
                zero_division=0
            )
            score2 = recall_score(
                y_sample,
                pred2[indices],
                zero_division=0
            )

        elif metric_name == "F1-score":
            score1 = f1_score(
                y_sample,
                pred1[indices],
                zero_division=0
            )
            score2 = f1_score(
                y_sample,
                pred2[indices],
                zero_division=0
            )

        elif metric_name == "ROC-AUC":
            score1 = roc_auc_score(
                y_sample,
                prob1[indices]
            )
            score2 = roc_auc_score(
                y_sample,
                prob2[indices]
            )

        elif metric_name == "PR-AUC":
            score1 = average_precision_score(
                y_sample,
                prob1[indices]
            )
            score2 = average_precision_score(
                y_sample,
                prob2[indices]
            )

        differences.append(score2 - score1)

    differences = np.array(differences)

    lower = np.percentile(differences, 2.5)
    upper = np.percentile(differences, 97.5)

    return lower, upper


# =========================
# DISPLAY COMPARISON
# =========================

print("\n")
print("=" * 55)
print("MODEL 1 VS MODEL 2 COMPARISON")
print("=" * 55)

print("\nModel 2 difference is calculated as:")
print("Random Forest - Logistic Regression")

print("\n")

for metric in metrics_1:

    difference = metrics_2[metric] - metrics_1[metric]

    lower, upper = bootstrap_metric_difference(
        y_test,
        y_pred_1,
        y_prob_1,
        y_pred_2,
        y_prob_2,
        metric,
        n_bootstrap=1000,
        random_state=42
    )

    print(f"{metric}:")
    print(f"  Model 1: {metrics_1[metric]:.4f}")
    print(f"  Model 2: {metrics_2[metric]:.4f}")
    print(f"  Difference: {difference:.4f}")
    print(
        f"  95% CI for difference: "
        f"{lower:.4f} - {upper:.4f}"
    )
    print()


print("Comparison calculations completed successfully.")