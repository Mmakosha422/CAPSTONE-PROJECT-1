import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. Set the folder containing the processed datasets
# ---------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# 2. Load the training and testing datasets
# ---------------------------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")


print("Training and testing data loaded successfully.")

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ---------------------------------------------------------
# 3. Convert age ranges into numerical age estimates
# ---------------------------------------------------------
# The original age variable is stored as ranges such as:
# [0-10), [10-20), [20-30), etc.
#
# We use the midpoint of each range as a numerical feature.

age_mapping = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95
}


X_train["age_numeric"] = X_train["age"].map(age_mapping)
X_test["age_numeric"] = X_test["age"].map(age_mapping)


# ---------------------------------------------------------
# 4. Create total prior hospital visits
# ---------------------------------------------------------
# Combines outpatient, emergency and inpatient visits
# recorded before the current encounter.

X_train["total_prior_visits"] = (
    X_train["number_outpatient"]
    + X_train["number_emergency"]
    + X_train["number_inpatient"]
)

X_test["total_prior_visits"] = (
    X_test["number_outpatient"]
    + X_test["number_emergency"]
    + X_test["number_inpatient"]
)


# ---------------------------------------------------------
# 5. Create a prior inpatient admission indicator
# ---------------------------------------------------------
# 1 = patient had at least one previous inpatient admission
# 0 = no previous inpatient admission

X_train["has_prior_inpatient"] = (
    X_train["number_inpatient"] > 0
).astype(int)

X_test["has_prior_inpatient"] = (
    X_test["number_inpatient"] > 0
).astype(int)


# ---------------------------------------------------------
# 6. Display the new features
# ---------------------------------------------------------

new_features = [
    "age_numeric",
    "total_prior_visits",
    "has_prior_inpatient"
]

print("\nNew engineered features:")
print(new_features)

print("\nTraining data with engineered features:")
print(X_train[new_features].head())


# ---------------------------------------------------------
# 7. Check for missing values in new features
# ---------------------------------------------------------

print("\nMissing values in engineered training features:")
print(X_train[new_features].isna().sum())

print("\nMissing values in engineered testing features:")
print(X_test[new_features].isna().sum())


# ---------------------------------------------------------
# 8. Save the feature-engineered datasets
# ---------------------------------------------------------

X_train.to_csv(
    DATA_DIR / "X_train_engineered.csv",
    index=False
)

X_test.to_csv(
    DATA_DIR / "X_test_engineered.csv",
    index=False
)


print("\nFeature engineering completed successfully.")

print("Saved:")
print("X_train_engineered.csv")
print("X_test_engineered.csv")