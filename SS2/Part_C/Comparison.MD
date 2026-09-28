# Model 1 vs Model 2 Comparison

## 1. Purpose

This document compares the performance of the two machine-learning classification models developed in Part B for predicting whether a patient will be readmitted to hospital within 30 days.

The two models evaluated are:

- **Model 1:** Logistic Regression
- **Model 2:** Random Forest

The comparison uses the same test dataset and the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

A paired bootstrap procedure with 1,000 resamples was used to estimate 95% confidence intervals for the difference between the two models.

The difference was calculated as:

**Random Forest − Logistic Regression**

---

## 2. Comparison Script

The comparison is performed using:

`comparison.py`

The script:

1. Loads the engineered training and test datasets produced in Part B.

2. Recreates the Logistic Regression pipeline.

3. Recreates the Random Forest pipeline.

4. Trains both models using the same training data.

5. Generates predictions and predicted probabilities on the test dataset.

6. Calculates the selected performance metrics.

7. Calculates the difference between the two models.

8. Uses paired bootstrap resampling with 1,000 resamples to calculate 95% confidence intervals for the differences.

###  run

From the project root directory, run:

```powershell
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_C\comparison.py