# SS2 Part B - Data Preparation and Modelling

## Project Introduction

In this project, factors associated with 30-day readmission were investigated using healthcare data and machine-learning classification techniques. 

The public dataset that was used for this proof-of-concept is the Diabetes 130-US Hospitals dataset.

The target variable is:

- `1` = readmitted within 30 days
- `0` = not readmitted within 30 days

## Project Workflow

The project follows these main steps:

1. **Data Preparation** – The raw dataset was loaded and cleaned
2. **Data Preprocessing** – Missing values were handled,  unsuitable variables were removed and the data for modelling was prepared.
3. **Feature Engineering** – additional features were created to help predict the 30-day hospital readmissions.
4. **Model 1: Logistic Regression** –  A Logistic Regression classification model was developed.
5. **Model 2: Random Forest** – A Random Forest classification model was developed.
6. **Model Evaluation** – The performance and evaluation of both models were compared.

## Running the Project
The Python scripts should be run in the following order:

1. `preprocessing.py`
2. `feature_engineering.py`
3. `model1.py`
4. `model2.py`

Make sure the required Python packages listed in `requirements.txt` are installed before running the scripts.

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

python SS2\Part_B\preprocessing.py



### Step 2- Feature Engineering

Run the feature engineering script:
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_B\feature_engineering.py

python SS2\Part_B\feature_engineering.py


### Step 3 - Model 1: Logistic Regression
Run the scripts in the following order.

& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_B\model1.py

python SS2\Part_B\model1.py

### Step 4 - Model 2: Random Forest
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_B\model2.py

python SS2\Part_B\model2.py