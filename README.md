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
pip install pandas numpy scikit-learn

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

## Dependencies

```
pandas
numpy
scikit-learn
```

