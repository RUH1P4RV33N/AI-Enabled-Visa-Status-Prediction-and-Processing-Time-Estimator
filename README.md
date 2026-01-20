# AI-Enabled Visa Status Prediction and Processing Time Estimator

## Overview

An intelligent web application built with **Streamlit** that predicts visa application outcomes and estimates processing times using advanced machine learning algorithms. The system analyzes 25+ data points including travel history, documentation quality, and applicant demographics to provide accurate predictions with confidence scores.

## 🌟 Features

- **Real-time Predictions**: Instant visa status prediction (Approved/Rejected)
- **Processing Time Estimation**: Accurate processing time predictions in days and weeks
- **Confidence Scoring**: Probability analysis for both approval and rejection
- **Interactive UI**: Modern, responsive design with smooth animations
- **Auto-training**: Automatically trains models on first run if not present
- **Comprehensive Analysis**: Processes travel history, documentation quality, and risk factors
- **High Scrutiny Detection**: Identifies sensitive country routes with strict processing

## 🚀 Live Demo

The application is deployed on **Streamlit Cloud** and accessible at:

https://visa-application-predictor.streamlit.app/


## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd visa-prediction-app
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📦 Required Dependencies

```txt
streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
scipy
joblib
```

## 🏃 Running the Application

### Local Development
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### First-Time Setup
On the first run, the system will:
1. Check for existing trained models
2. If not found, automatically train models using the training dataset
3. Save models to the `models/` directory
4. Display training progress and metrics

## 📁 Project Structure

```
visa-prediction-app/
│
├── app.py                          # Main Streamlit application
├── src/
│   ├── main.py
|   ├── eda_outputs/                # EDA visualizations (auto-generated)
│   └── data_analysis.py            # EDA and visualization functions
│
├── data/
│   └── train.csv                   # Training dataset
│
├── models/                         # Saved ML models (auto-generated)
│   ├── visa_model_status.pkl
│   ├── visa_model_time.pkl
│   ├── visa_model_encoders.pkl
│   └── visa_model_features.pkl
│
├── public/
│   └── images/
│       ├── background.jpg          # Background image
│       ├── heading-image.jpg       # Header image
│       └── logo.png               # App logo
│                   
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🎯 How to Use

### 1. Personal Information
- Select **Applicant Country** and **Destination Country**
- Enter **Age** (18-80 years)
- Select **Gender**, **Visa Category**, and **Employment Status**
- Choose **Marital Status** and **Application Type**

### 2. Travel History
- Enter number of **Countries Visited**
- Specify visits to **Schengen**, **USA**, and **UK**
- Enter **Previous Rejections** count
- Indicate **Overstay History** (Yes/No)

### 3. Documentation & Processing
- Confirm **Document Completeness**
- Indicate availability of **Supporting Documents**
- Specify **Financial Documents** status
- Select **Biometrics Completion** and **Priority Processing**
- Choose **Processing Office**

### 4. Get Prediction
- Click the **"GET PREDICTION"** button
- View detailed results including:
  - Predicted Status (Approved/Rejected)
  - Confidence Score
  - Processing Time Estimate
  - Probability Analysis

## 🤖 Machine Learning Models

### Status Classification
The system evaluates three ensemble models and selects the best performer:
- **Random Forest Classifier** (200 estimators, max_depth=20)
- **XGBoost Classifier** (200 estimators, max_depth=10)
- **Gradient Boosting Classifier** (150 estimators, max_depth=8)

**Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Feature importance analysis
- Classification report

### Processing Time Regression
Predicts visa processing time in days:
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Gradient Boosting Regressor**

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

## 🔍 Feature Engineering

### Composite Features
1. **Travel Score** (0-1 scale)
   ```
   (countries_visited × 2 + schengen_visits × 4 + us_visits × 5 + uk_visits × 3) / 95
   ```

2. **Documentation Quality Score** (0-1 scale)
   ```
   document_completeness × 0.4 + supporting_docs × 0.3 + financial_docs × 0.3
   ```

3. **Risk Score** (0-1 scale)
   ```
   overstay_history × 0.35 + previous_rejections × 0.2 + 
   incomplete_docs × 0.2 + missing_support_docs × 0.15 + 
   missing_financial_docs × 0.15 + high_scrutiny_penalty × 0.25
   ```

### Categorical Features
- **Age Groups**: Minor, Young_Adult, Adult, Middle_Aged, Senior
- **Travel Experience**: None, Limited, Moderate, Extensive

### High Scrutiny Routes

The system flags **34 country pairs** for enhanced processing and automatic rejection:

**USA Routes:**
- Afghanistan, Pakistan, Bangladesh, Nigeria, Somalia, Yemen, Syria, Iraq, Iran, Libya, Sudan → USA

**UK Routes:**
- Afghanistan, Pakistan, Bangladesh, Nigeria, Somalia, Iraq, Syria, Yemen → UK

**Schengen Routes:**
- Afghanistan, Pakistan, Bangladesh, Nigeria, Somalia, Sudan → Schengen countries

**Canada Routes:**
- Afghanistan, Pakistan, Bangladesh, Nigeria, Iran → Canada

**Australia Routes:**
- Afghanistan, Pakistan, Bangladesh, Nigeria → Australia

**Note:** Applications from these routes with incomplete documentation receive automatic rejection with explanation.

## 📊 Model Training

### Automatic Training
```python
# On first run, the app automatically:
# 1. Loads data from data/train.csv
# 2. Engineers features
# 3. Trains both models
# 4. Saves to models/ directory
```

### Manual Training
```bash
cd src
python main.py
```

This will:
- Load and preprocess training data
- Engineer all features
- Train status and time models
- Generate EDA visualizations in `eda_outputs/`
- Save models to `models/` directory
- Display performance metrics

## 🎨 UI Features

### Modern Design
- Glassmorphism sidebar with semi-transparent background
- Gradient buttons with hover effects
- Logical grouping of fields to reduce cognitive load
- Responsive metric cards with hover animations
- Dynamic success and warning highlights based on prediction outcomes

### User Experience
- **Reset Functionality**: One-click reset clears all inputs, predictions, and UI states for a fresh start
- **User-Friendly Design**: No technical knowledge required to use the application
- **Scalable UI Structure**: Easy to extend with additional features or models
- **Instant Prediction Feedback**: Results rendered immediately after model inference
- **Responsive Design**: Fully responsive layout that adapts seamlessly across desktops, tablets, and different screen resolutions

## 📈 Input Data Requirements

### Required CSV Columns
- `submission_date`, `decision_date` (DD-MM-YYYY format)
- `applicant_country`, `destination_country`, `visa_category`
- `countries_visited`, `schengen_visits`, `us_visits`, `uk_visits`
- `applicant_age`, `previous_rejections`
- Binary columns: `priority_processing`, `biometrics_completed`, `overstay_history`, etc.
- Categorical: `season`, `gender`, `employment_status`, `marital_status`, `processing_office`

## 🔮 Prediction API

### VisaPredictionSystem Class

```python
from main import VisaPredictionSystem

# Initialize system
system = VisaPredictionSystem()

# Load trained models
system.load_models("visa_model")

# Make prediction
application_data = {
    "applicant_country": "India",
    "destination_country": "USA",
    "visa_category": "Tourist",
    # ... other fields
}

result = system.predict(application_data)
```

### Prediction Output
```python
{
    "predicted_status": "Approved" or "Rejected",
    "approval_probability": float (0-1),
    "rejection_probability": float (0-1),
    "estimated_processing_days": int,
    "estimated_processing_weeks": float,
    "high_scrutiny_route": bool
}
```

## 📊 Exploratory Data Analysis

The system generates 8 comprehensive visualization sets:

1. **Target Distribution**: Application status pie chart, processing time histogram
2. **Visa Category Analysis**: Application counts, approval rates, processing times
3. **Geographic Analysis**: Top countries, approval rates by country
4. **Correlation Analysis**: Feature correlation matrix
5. **Travel History Impact**: Travel experience vs approval rates
6. **Document Quality Analysis**: Documentation impact on outcomes
7. **Seasonal Patterns**: Monthly trends, seasonal approval rates
8. **Demographics Analysis**: Age distribution, employment status impact

Run `perform_comprehensive_eda(df)` to generate all visualizations.



## 📝 Disclaimer

⚠️ **Important**: This tool provides AI-generated predictions for educational and informational purposes only. Actual visa decisions are made by immigration authorities and depend on numerous factors beyond this model's scope. Always consult official sources and qualified immigration professionals for visa-related decisions.



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.




---

**Built with ❤️ using Python, Streamlit, and ML libraries**