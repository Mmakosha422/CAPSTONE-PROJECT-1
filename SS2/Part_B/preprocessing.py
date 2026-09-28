import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------
# 1. Load the original dataset
# ---------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent / "diabetic_data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)


# ---------------------------------------------------------
# 2. Replace '?' with missing values
# ---------------------------------------------------------

df = df.replace("?", pd.NA)


# ---------------------------------------------------------
# 3. Remove columns with a very high proportion of
#    missing values
# ---------------------------------------------------------

columns_to_drop = [
    "weight",
    "max_glu_serum",
    "A1Cresult",
    "medical_specialty",
    "payer_code"
]

df = df.drop(columns=columns_to_drop)

print("\nColumns removed because of high missingness:")
print(columns_to_drop)


# ---------------------------------------------------------
# 4. Remove identifier columns
# ---------------------------------------------------------

identifier_columns = [
    "encounter_id",
    "patient_nbr"
]

df = df.drop(columns=identifier_columns)

print("\nIdentifier columns removed:")
print(identifier_columns)


# ---------------------------------------------------------
# 5. Create the binary target
#    1 = readmitted within 30 days
#    0 = not readmitted within 30 days
# ---------------------------------------------------------

df["readmitted_within_30_days"] = df["readmitted"].map({
    "<30": 1,
    ">30": 0,
    "NO": 0
})


# ---------------------------------------------------------
# 6. Separate features and target
# ---------------------------------------------------------

X = df.drop(
    columns=["readmitted", "readmitted_within_30_days"]
)

y = df["readmitted_within_30_days"]


# ---------------------------------------------------------
# 7. Split the data BEFORE fitting imputers
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ---------------------------------------------------------
# 8. Identify categorical and numerical columns
# ---------------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()


print("\nNumber of categorical columns:",
      len(categorical_columns))

print("Number of numerical columns:",
      len(numerical_columns))


# ---------------------------------------------------------
# 9. Impute missing values
#    IMPORTANT: imputers are fitted ONLY on training data
# ---------------------------------------------------------

categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

numerical_imputer = SimpleImputer(
    strategy="median"
)


X_train[categorical_columns] = categorical_imputer.fit_transform(
    X_train[categorical_columns]
)

X_test[categorical_columns] = categorical_imputer.transform(
    X_test[categorical_columns]
)


X_train[numerical_columns] = numerical_imputer.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = numerical_imputer.transform(
    X_test[numerical_columns]
)


# ---------------------------------------------------------
# 10. Check for remaining missing values
# ---------------------------------------------------------

print("\nMissing values in X_train:",
      X_train.isna().sum().sum())

print("Missing values in X_test:",
      X_test.isna().sum().sum())


# ---------------------------------------------------------
# 11. Save the processed train/test datasets
# ---------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent

X_train.to_csv(
    OUTPUT_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    OUTPUT_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


print("\nProcessed datasets saved successfully.")

print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")