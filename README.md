# STADIOcare Data Science Capstone Project

**Client:** STADIOcare

## 1. Project Motivation - PART A
STADIOcare Group is one of the leading private healthcare organisations in South Africa, that provides essential healthcare services nationwide. These primarily include acute hospital care, day surgery centres, emergency medical services, primary care clinics and mental health services.
Although STADIOcare group operates under a simple promise "the right care, in the right place, at the right time." Howvever, one of the major challanges that the  organisation faces is an increasing rate of patients readmission within 30 days of discharge. According to the STADIOcare brief pack, the 30 day readamission rate increased from  8.9% to 11.3% and at the same time time, the average stay length increased from 3.9 to 4.2 nights. This is becoming a pressing issue because the cost of hospital resources and patient care is getting alot harder. 



The STADIOcare 2030 Strategy sets five priorities for the next five years:

- Fill the beds we have, safely.
- Keep patients out of hospital when they should be.
- Protect the margin.
- Make Serene the market leader in mental health.
- Become a data-led business.


Of these priorities, the second is particularly relevant to this project because it focuses on supporting patients who are likely to return to hospital within a month. STADIOcare provides healthcare data such as admission, discharge and transfer data, including patient movements, wards, procedure codes, and discharge status, with five years of historical data available. Using this data to analyse and investigate this challenge will provide a clear understanding of patterns related to patients who are readmitted within 30 days of discharge. 


In conclusion, the the Capstone project is to investigate and understand the main factors associated with 30 day patient readmission and investigate whether relevant healthcare data can be used to identify which patients are more likely to be readmitted within 30 days of discharge. With this type of insight and understanding, patient management and decision-making will make STADIOcare system run much more effectively and smoothly while ensuring each patient recieves the best healthcare treatment. 

## 2. Problem Statement - PART B
Although STADIOcare aims to provide "the right care, in the right place, at the right time,” The organisation has still seen an increase in patient readmission rate within 30 days, from 8.9% to 11.3%. Despite all the promises, it still remains unclear which factors are associated with patients being readmitted within 30 days after discharge. Therefore, the purpose of the study is to investigate these factors and investigate healthcare  data that can be possible used to identify which patients are most likely to be readmitted within 30 days of discharge.


## 3. Repository Structure - PART D

This repository is organised to clearly separate the datasets, models,
experimental setup, experimental results, statistical helper and
comparison scripts, and visualisation materials used in the project.

### Dataset
Contains datasets and data-related documentation, including the Part C
Data Requirements document.

### Models
Contains machine learning models developed for the project.

### Experimental setup
Contains files and documentation describing the experimental setup.

### Experimental results
Contains results generated from experiments and model evaluation.

### Statistical helper and comparison scripts
Contains statistical helper functions and scripts used for analysis and
comparison.

### Visualisation
Contains scripts and files used to create visualisations.


## RAAIDD Log (Part E)


| RAAIDD | Description |
|---|---|
|**Risks** |The STADIOcare organisation may contain incorrect records and also missing data which may result in incorrect interpretations and analysis down the line. Since STADIOcare primarily contains sensitive data, incorrect interpretations or analysis could possibly result in confidentiality risks. Lastly, Differences in data quality across STADIOcare facilities could make it difficult to identify reliable and consistent patterns associated with 30-day patient readmissions. |
| **Actions** | The Data cleaning process is required to ensure consistance throught. Data capturers and healthcare workers must ensure that patient admission, discharge and readmission information is recorded accurately and consistently in the relevant systems. All the data that is analysed and collected will be protected and  remain private.  |
| **Assumptions** | Assuming that STADIOcare will be able to provide enough data that is related to the problem stataement. Assuming that patient admission and discharge dates are recorded daily and accurately in the relevant healthcare systems. Assuming all data that is provided is accurate and has not been fabricated.|
| **Issues** | Errors in capturing patients  information may result in inaccurate admission, discharge or readmission records, affecting the accuracy of the 30-day readmission analysis.Missing clinical information may cause an issue down the line when identifying important factors that are related to readmission.The dataset provided might be is too little or complex to analyse. |
| **Decisions** | The main goal is to stick to basic machine learning models since dataset to make things much more simpler.The patient information that will be used will remain anonymise to protect the privacy of each patient.The data quality will be assessed and addressed before developing any machine learning models. |
| **Dependencies** |The historical data from STADIOcare is needed before any steps can occur. Dataset from other facilities are required in oreder to be able to compare and analyse 30 day readmission patterns  and identify the gap. The usage of other external datasets like kaggle and datacamp will be helpful with comparison to the STADIOcare dataset. The cleaned dataset will be available before exploratory data analysis and visualisations can be produced. |
