# Model 2 Performance – Random Forest

## 1. Purpose

This document presents the performance results of Model 2, the Random Forest classification model developed to predict whether a patient will be readmitted to hospital within 30 days.

The model was evaluated using the same test dataset used for Model 1.

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

`model2_performance.py`

The script:

1. Loads the engineered training and test datasets.

2. Recreates the Random Forest preprocessing pipeline.

3. Trains the Random Forest model.

4. Generates predictions on the test dataset.

5. Calculates the selected performance metrics.

6. Calculates the confusion matrix.

7. Uses 1,000 bootstrap resamples to estimate 95% confidence intervals.

### To run

From the project root directory:

```powershell
& "$env:USERPROFILE\anaconda3\envs\ds182\python.exe" SS2\Part_C\model2_performance.py