# Model 1 Performance – Logistic Regression

## 1. Purpose

This document presents the performance results of Model 1, the Logistic Regression classification model developed to predict whether a patient will be readmitted to hospital within 30 days.

The model was evaluated using the test dataset prepared during Part B.

The following performance metrics were calculated:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

A bootstrap procedure with 1,000 resamples was also used to calculate 95% confidence intervals for the performance metrics.

---

## 2. Performance Script

The performance calculations are performed using:

`model1_performance.py`

The script:

1. Loads the engineered training and test datasets.
2. Recreates the Logistic Regression preprocessing pipeline.
3. Trains the Logistic Regression model.
4. Generates predictions on the test dataset.
5. Calculates the selected performance metrics.
6. Calculates the confusion matrix.
7. Uses 1,000 bootstrap resamples to estimate 95% confidence intervals.

### How to run

From the project root directory:

```powershell
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_C\model1_performance.py