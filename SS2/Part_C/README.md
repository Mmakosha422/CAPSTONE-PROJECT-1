# SS2 Part C – Model Performance and Comparison

## 1. Overview

Part C presents the results obtained from applying the two machine-learning classification models developed in Part B to the public diabetes hospital readmission dataset.

The objective is to evaluate and compare the performance of:

- **Model 1:** Logistic Regression
- **Model 2:** Random Forest

Both models predict whether a patient will be readmitted to hospital within 30 days.

The models are evaluated using multiple classification metrics and statistical analysis.

---

## 2. Dataset

The analysis uses the engineered training and test datasets produced during Part B.

The test dataset contains:

- **20,354 observations**
- **45 engineered features**

The same test dataset is used for both models to ensure that their performance can be compared on the same observations.

The original public dataset is the **Diabetes 130-US Hospitals for Years 1999-2008** dataset.

---

## 3. Part C Files

| File | Description |
|---|---|
| `Model1Performance.MD` | Documents the performance of the Logistic Regression model |
| `Model2Performance.MD` | Documents the performance of the Random Forest model |
| `Comparison.MD` | Compares the performance of both models |
| `model1_performance.py` | Calculates Model 1 performance metrics and bootstrap confidence intervals |
| `model2_performance.py` | Calculates Model 2 performance metrics and bootstrap confidence intervals |
| `comparison.py` | Calculates and compares the performance of both models |
| `requirements.txt` | Lists the Python packages required to run the Part C scripts |

---

## 4. Model 1 Performance

The detailed results for Logistic Regression are available in:

[Model 1 Performance](Model1Performance.MD)

The calculations are performed using:

`model1_performance.py`

### Run command

```powershell
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_C\model1_performance.py