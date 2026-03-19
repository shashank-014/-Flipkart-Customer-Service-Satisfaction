# Flipkart Customer Satisfaction Prediction

This folder contains a Streamlit project for analyzing Flipkart support interactions and predicting customer satisfaction from the service data.

## What This Project Does

- Loads and cleans the support dataset from `Customer_support_data.csv`
- Converts CSAT into a binary satisfaction target
- Builds a support-focused EDA story
- Compares multiple classification models
- Shows the best model, feature importance, and a simple prediction form

## Folder Layout

- `app.py` - Streamlit app for the project
- `Customer_support_data.csv` - source dataset
- `Customer_Service_Satisfaction_Prediction_for_Flipkart_Using_Machine_Learning.ipynb` - original notebook

## App Walkthrough

The app is laid out in the same order I would present it:

1. Overview for the project story and key stats.
2. Data understanding for the raw support file.
3. Data wrangling for the cleanup and label setup.
4. EDA for the visual patterns.
5. Hypothesis testing for the timing checks.
6. Feature engineering for the usable model inputs.
7. Modeling for the comparison table and final choice.
8. Conclusion for the short business takeaway.

## Project Style

The app and docs keep the same tone as the notebook:
- simple, direct language
- business-first framing

## How To Run

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the app

```powershell
streamlit run app.py
```

## Data Notes

- Rows: 85,907
- Columns: 20
- Target: `CSAT Score`
- Satisfaction label: `CSAT >= 4` means satisfied
- Missing values: present in several operational fields
- Duplicate rows: none

## Modeling Notes

The app compares:

- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost, if available in the environment

The split is stratified because the target is imbalanced.
