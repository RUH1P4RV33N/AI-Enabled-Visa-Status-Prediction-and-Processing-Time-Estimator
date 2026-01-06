# AI Enabled Visa Status Prediction and Processing Time Estimator

## Overview

This module handles data loading, preprocessing, and feature engineering for visa application data. It transforms raw CSV data into ML-ready features.

## Features

- Date parsing and temporal feature extraction
- Composite feature engineering (travel score, risk score, doc quality)
- Categorical encoding with Label Encoders
- Binary feature conversion

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing
cd src
python main.py
```

## Functions

### `load_and_preprocess_data(filepath)`
Loads CSV and extracts temporal features.

**Input:** Path to CSV file  
**Output:** Preprocessed DataFrame with `processing_days`, `submission_month`, `submission_quarter`

### `engineer_features(df)`
Creates composite features:
- `travel_score`: Weighted travel history (0-1 scale)
- `doc_quality_score`: Documentation completeness (0-1 scale)
- `risk_score`: Risk assessment (0-1 scale)
- `age_group`: Categorical age bins
- `travel_experience`: Travel experience level
- `high_scrutiny_route`: Binary flag for sensitive routes

### `prepare_features_for_modeling(df)`
Encodes categorical variables for ML models.

**Output:** 
- `df_model`: Encoded DataFrame
- `label_encoders`: Dictionary of LabelEncoder objects

prepare_features_for_modeling(df)
Prepares data for machine learning models.
Encoding Strategy:

Binary columns: Converted to 0/1 integers
Categorical columns: Label encoded with _encoded suffix
Numerical columns: Kept as-is

Output:

df_model: Fully encoded DataFrame
feature_cols: List of all feature column names for modeling
label_encoders: Dictionary of LabelEncoder objects for decoding

data_analysis.py
perform_comprehensive_eda(df, output_dir="eda_outputs")
Generates 8 comprehensive visualization sets covering:

Target Distribution (01_target_distribution.png)

Application status pie chart
Processing time histogram with mean line


Visa Category Analysis (02_visa_category_analysis.png)

Application counts by category
Approval rates by category
Processing time boxplots
Category vs Season heatmap


Geographic Analysis (03_geographic_analysis.png)

Top 10 applicant countries
Top 10 destination countries
Country-wise approval rates
Destination-wise processing times


Correlation Analysis (04_correlation_analysis.png)

Full feature correlation matrix
Processing time correlation bar chart


Travel History Impact (05_travel_history_impact.png)

Travel experience vs approval
Previous rejections impact
Overstay history impact
Travel score distributions


Document Quality Analysis (06_document_quality_analysis.png)

Document completeness impact
Financial documentation impact
Quality score vs processing time scatter
Risk score distributions


Seasonal Patterns (07_seasonal_patterns.png)

Applications by season (pie chart)
Processing time by season
Monthly application trends
Seasonal approval rates


Demographics Analysis (08_demographics_analysis.png)

Age distribution histogram
Age group approval rates
Employment status impact
Marital status impact



Output: All visualizations saved to output_dir with 300 DPI quality

## Input Data Requirements

CSV file with columns:
- `submission_date`, `decision_date` (DD-MM-YYYY format)
- `applicant_country`, `destination_country`, `visa_category`
- `countries_visited`, `schengen_visits`, `us_visits`, `uk_visits`
- `applicant_age`, `previous_rejections`
- Binary columns: `priority_processing`, `biometrics_completed`, `overstay_history`, etc.
- Categorical: `season`, `gender`, `employment_status`, `marital_status`, `processing_office`

## Output Features

**Adding Features:**
- `travel_score` = (countries×2 + schengen×4 + us×5 + uk×3) / 95
- `doc_quality_score` = (complete×0.4 + supporting×0.3 + financial×0.3)
- `risk_score` = (overstay×0.3 + rejections×0.15 + incomplete_docs×0.25 + ...)

**Encoded Columns:**
All categorical columns get `_encoded` suffix (e.g., `applicant_country_encoded`)

## High Scrutiny Routes

The module flags 22 country pairs for enhanced processing:
- India/China/Pakistan → USA
- Syria/Afghanistan/Iraq → Germany/UK
- Nigeria → USA/UK
- And more...

## Model Training and Prediction

### Visa Status Classification

The system trains ensemble classifiers to predict visa application outcomes:

**Models Evaluated:**
- Random Forest Classifier (200 estimators, max_depth=20)
- XGBoost Classifier (200 estimators, max_depth=10)
- Gradient Boosting (150 estimators, max_depth=8)

**Output:**
- Best performing model selected based on accuracy
- Classification report with precision, recall, F1-score
- Top 10 most important features for prediction

### Processing Time Regression

Predicts estimated processing time in days:

**Models Evaluated:**
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor

**Metrics:**
- MAE (Mean Absolute Error) in days
- RMSE (Root Mean Squared Error)
- R² Score
- Top 10 features affecting processing time

### VisaPredictionSystem Class

A complete end-to-end prediction system for visa applications.

#### Methods

**`__init__()`**
Initializes empty prediction system with placeholders for models and encoders.

**`train(df)`**
Trains both status classifier and processing time regressor.

**Input:** Preprocessed DataFrame with all features  
**Output:** 
- `status_accuracy`: Classification accuracy score
- `time_mae`: Mean Absolute Error for processing time prediction

**Process:**
- Splits data 80/20 train/test
- Trains status classifier (approval/rejection)
- Trains processing time regressor
- Reports performance metrics

**`save_models(prefix="visa_model")`**
Persists trained models to disk.

**Output Files:**
- `{prefix}_status.pkl`: Status classification model
- `{prefix}_time.pkl`: Processing time regression model
- `{prefix}_encoders.pkl`: Label encoders dictionary
- `{prefix}_features.pkl`: Feature column list

**`load_models(prefix="visa_model")`**
Loads previously trained models from disk.

**`predict(application_data)`**
Makes predictions for new visa applications.

**Input:** Dictionary or DataFrame with application details  
**Output:** Dictionary containing:
```python
{
    "predicted_status": "Approved" or "Rejected",
    "approval_probability": float (0-1),
    "rejection_probability": float (0-1),
    "estimated_processing_days": int,
    "estimated_processing_weeks": float
}
```

**Required Input Fields:**
- `applicant_country`, `destination_country`
- `application_type`, `visa_category`, `season`
- `priority_processing`, `biometrics_completed`, `overstay_history`
- `countries_visited`, `schengen_visits`, `us_visits`, `uk_visits`
- `previous_rejections`, `applicant_age`
- `gender`, `employment_status`, `marital_status`
- `document_completeness`, `supporting_docs_provided`
- `interview_required`, `financial_docs_provided`, `sponsorship_letter`
- `processing_office`
- `submission_month`, `submission_day_of_week`, `submission_quarter`

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
scipy
joblib
```

