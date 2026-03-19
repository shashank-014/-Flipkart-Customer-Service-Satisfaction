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
- `project_notebook_code_export.py` - notebook export used as a reference
- `Customer_Service_Satisfaction_Prediction_for_Flipkart_Using_Machine_Learning.ipynb` - original notebook
- `Project_Presentation_Script_20min.md` - updated presentation read-out
- `requirements.txt` - Python dependencies
- `DELETED_FILES/` - archived older docs from this folder

## Project Style

The app and docs keep the same tone as the notebook:

- first-person explanations
- simple, direct language
- business-first framing
- short functional code comments

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

## Presentation Notes

The updated presentation read-out is in `Project_Presentation_Script_20min.md`.

## History

See `history.md` for the latest update log inside this folder.
