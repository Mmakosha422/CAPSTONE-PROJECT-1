# SS2 Part B - Machine Learning Models

## Project Introduction

This project investigates factors associated with 30-day hospital readmission using healthcare data and machine-learning classification techniques.

The public dataset used for this proof-of-concept is the Diabetes 130-US Hospitals dataset.

The target variable is:

- `1` = readmitted within 30 days
- `0` = not readmitted within 30 days

## Part B Workflow

## The project follows the following workflow:

```text
Raw Dataset
     ↓
Preprocessing
     ↓
Feature Engineering
     ↓
Model 1: Logistic Regression
     ↓
Model 2: Random Forest
     ↓
Model Evaluation

Running the Project

The scripts were developed and tested using the Python environment: ds182

The scripts should be executed in the following order:


## Running the Project

The scripts were developed and tested using the `ds182` Python environment with Python 3.11.15.

Run the scripts in the following order.

### Step 1 - Preprocessing

Run the preprocessing script:

```powershell
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_B\preprocessing.py

### Step 2- Feature Engineering

Run the feature engineering script:
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_B\feature_engineering.py

### Step 3 - Model 1: Logistic Regression
Run the scripts in the following order.

& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_B\model1.py

### Step 4 - Model 2: Random Forest
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_B\model2.py