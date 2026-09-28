import pandas as pd
from pathlib import Path


# Load the original dataset
DATA_PATH = Path(__file__).resolve().parent / "diabetic_data.csv"

df = pd.read_csv(DATA_PATH)

# Replace '?' with missing values
df = df.replace("?", pd.NA)

# Display basic information
print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum())

# Remove columns with a very high proportion of missing values
columns_to_drop = [
    "weight",
    "max_glu_serum",
    "A1Cresult",
    "medical_specialty",
    "payer_code"
]

df = df.drop(columns=columns_to_drop)

# Fill missing categorical values with the most frequent value
categorical_columns = ["race", "diag_1", "diag_2", "diag_3"]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


print("\nRemaining missing values:")
print(df.isna().sum().sum())

print("\nColumns removed:")
print(columns_to_drop)

print("\nNew dataset shape:")
print(df.shape)

# Remove identifier columns
identifier_columns = ["encounter_id", "patient_nbr"]

df = df.drop(columns=identifier_columns)

print("\nIdentifier columns removed:")
print(identifier_columns)

print("\nDataset shape after removing identifiers:")
print(df.shape)


# Save the cleaned dataset
OUTPUT_PATH = Path(__file__).resolve().parent / "cleaned_data.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully.")
print("Saved to:", OUTPUT_PATH)