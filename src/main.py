import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_analysis import perform_comprehensive_eda

# Visualization Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib

# Set visualization style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Route complexity indicator
high_scrutiny_routes = [
    # USA
    ("Afghanistan", "USA"),
    ("Pakistan", "USA"),
    ("Bangladesh", "USA"),
    ("Nigeria", "USA"),
    ("Somalia", "USA"),
    ("Yemen", "USA"),
    ("Syria", "USA"),
    ("Iraq", "USA"),
    ("Iran", "USA"),
    ("Libya", "USA"),
    ("Sudan", "USA"),
    # UK
    ("Afghanistan", "UK"),
    ("Pakistan", "UK"),
    ("Bangladesh", "UK"),
    ("Nigeria", "UK"),
    ("Somalia", "UK"),
    ("Iraq", "UK"),
    ("Syria", "UK"),
    ("Yemen", "UK"),
    # Schengen (generalized as destination)
    ("Afghanistan", "Schengen"),
    ("Pakistan", "Schengen"),
    ("Bangladesh", "Schengen"),
    ("Nigeria", "Schengen"),
    ("Somalia", "Schengen"),
    ("Sudan", "Schengen"),
    # Canada
    ("Afghanistan", "Canada"),
    ("Pakistan", "Canada"),
    ("Bangladesh", "Canada"),
    ("Nigeria", "Canada"),
    ("Iran", "Canada"),
    # Australia
    ("Afghanistan", "Australia"),
    ("Pakistan", "Australia"),
    ("Bangladesh", "Australia"),
    ("Nigeria", "Australia"),
]


def load_and_preprocess_data(filepath):
    print(f"\nLoading data...")
    df = pd.read_csv(filepath)
    print(f"\nLoaded {len(df)} records with {len(df.columns)} features")

    # Converting string dates to proper datetime format
    df["submission_date"] = pd.to_datetime(df["submission_date"], format="%d-%m-%Y")
    df["decision_date"] = pd.to_datetime(df["decision_date"], format="%d-%m-%Y")

    # Calculating actual processing time
    df["processing_days"] = (df["decision_date"] - df["submission_date"]).dt.days

    # Basic date related features
    df["submission_month"] = df["submission_date"].dt.month
    df["submission_day_of_week"] = df["submission_date"].dt.dayofweek
    df["submission_quarter"] = df["submission_date"].dt.quarter

    return df


def engineer_features(df):
    """Create advanced features for better predictions"""
    print("\n🔧 Engineering features...")

    # Travel history score - normalized properly
    max_realistic_travel = 15 * 2 + 5 * 4 + 3 * 5 + 3 * 3  # 74
    df["travel_score"] = (
        df["countries_visited"] * 2
        + df["schengen_visits"] * 4
        + df["us_visits"] * 5
        + df["uk_visits"] * 3
    ) / max_realistic_travel
    df["travel_score"] = df["travel_score"].clip(0, 1)

    # Documentation quality score
    df["doc_quality_score"] = (
        (df["document_completeness"] == "Yes").astype(int) * 0.4
        + (df["supporting_docs_provided"] == "Yes").astype(int) * 0.3
        + (df["financial_docs_provided"] == "Yes").astype(int) * 0.3
    )

    # Age groups
    df["age_group"] = pd.cut(
        df["applicant_age"],
        bins=[0, 18, 30, 45, 60, 100],
        labels=["Minor", "Young_Adult", "Adult", "Middle_Aged", "Senior"],
    )

    # Travel experience level
    df["travel_experience"] = pd.cut(
        df["countries_visited"],
        bins=[-1, 0, 5, 15, 50],
        labels=["None", "Limited", "Moderate", "Extensive"],
    )

    df["high_scrutiny_route"] = df.apply(
        lambda x: 1
        if (x["applicant_country"], x["destination_country"]) in high_scrutiny_routes
        else 0,
        axis=1,
    )

    # Risk factors - HEAVILY WEIGHTED for high scrutiny routes with weak documentation
    # High scrutiny route + incomplete docs or red flags = very high risk
    df["risk_score"] = (
        (df["overstay_history"] == "Yes").astype(int) * 0.35
        + (df["previous_rejections"] * 0.2).clip(0, 0.4)
        + (df["document_completeness"] == "No").astype(int) * 0.2
        + (df["supporting_docs_provided"] == "No").astype(int) * 0.15
        + (df["financial_docs_provided"] == "No").astype(int) * 0.15
        # HIGH SCRUTINY PENALTY: If on scrutiny route + ANY documentation issue
        + (
            df["high_scrutiny_route"]
            * (
                (df["document_completeness"] == "No").astype(int)
                + (df["supporting_docs_provided"] == "No").astype(int)
                + (df["financial_docs_provided"] == "No").astype(int)
            )
            * 0.25  # Major penalty multiplier
        )
    )
    df["risk_score"] = df["risk_score"].clip(0, 1)  # Cap at 1.0

    print(f"✓ Created {len(df.columns)} total features")
    return df


def prepare_features_for_modeling(df):
    print("\nPreparing features for modeling...")

    # Categorical columns to encode
    categorical_cols = [
        "applicant_country",
        "destination_country",
        "application_type",
        "visa_category",
        "season",
        "gender",
        "employment_status",
        "marital_status",
        "processing_office",
        "age_group",
        "travel_experience",
    ]

    # Binary columns
    binary_cols = [
        "priority_processing",
        "biometrics_completed",
        "overstay_history",
        "document_completeness",
        "supporting_docs_provided",
        "interview_required",
        "financial_docs_provided",
        "sponsorship_letter",
    ]

    # Numerical columns
    numerical_cols = [
        "applicant_age",
        "countries_visited",
        "schengen_visits",
        "us_visits",
        "uk_visits",
        "previous_rejections",
        "submission_month",
        "submission_day_of_week",
        "submission_quarter",
        "travel_score",
        "doc_quality_score",
        "risk_score",
        "high_scrutiny_route",
    ]

    # Create a copy to avoid modifying original dataframe
    df_model = df.copy()

    # Convert binary columns to 0/1
    for col in binary_cols:
        df_model[col] = (df_model[col] == "Yes").astype(int)

    # Label Encoding for categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col + "_encoded"] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le

    # Select feature columns
    feature_cols = (
        [col + "_encoded" for col in categorical_cols] + binary_cols + numerical_cols
    )

    return df_model, feature_cols, label_encoders


# ============================================================================
# 2. MODEL TRAINING - VISA STATUS CLASSIFICATION
# ============================================================================


def train_status_classifier(X_train, y_train, X_test, y_test):
    """Train ensemble model for visa status prediction"""
    print("\nTRAINING VISA STATUS CLASSIFIER\n")

    # Try multiple models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42
        ),
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        if name != "Gradient Boosting":  # Skip GB for classification
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"   Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name

    print(f"\nBest Model: {best_name} with accuracy {best_accuracy:.4f}")

    # Detailed evaluation
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        print("\nTop 10 Most Important Features:")
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": best_model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )
        for idx, row in feature_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

    return best_model, best_accuracy


# ============================================================================
# 3. MODEL TRAINING - PROCESSING TIME REGRESSION
# ============================================================================


def train_processing_time_regressor(X_train, y_train, X_test, y_test):
    print("\nTRAINING PROCESSING TIME REGRESSOR\n")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42
        ),
    }

    best_model = None
    best_mae = float("inf")
    best_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"   MAE: {mae:.2f} days")
        print(f"   RMSE: {rmse:.2f} days")
        print(f"   R² Score: {r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name} with MAE {best_mae:.2f} days")

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        print("\nTop 10 Most Important Features for Processing Time:")
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": best_model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )
        for idx, row in feature_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

    return best_model, best_mae


# ============================================================================
# 4. MODEL PERSISTENCE AND PREDICTION
# ============================================================================


class VisaPredictionSystem:
    """Complete visa prediction system"""

    def __init__(self):
        self.status_model = None
        self.time_model = None
        self.label_encoders = {}
        self.feature_cols = []
        self.scaler = StandardScaler()

    def train(self, df):
        print("\nTRAINING COMPLETE PREDICTION SYSTEM\n")

        # Prepare features
        df_model, self.feature_cols, self.label_encoders = (
            prepare_features_for_modeling(df)
        )

        # Prepare data for status classification
        X_status = df_model[self.feature_cols]
        y_status = (df_model["application_status"] == "Approved").astype(int)

        # Prepare data for processing time regression
        X_time = df_model[self.feature_cols]
        y_time = df_model["processing_days"]

        # Train-test split with detailed logging
        print("\nSplitting data for training and testing...")
        X_train_status, X_test_status, y_train_status, y_test_status = train_test_split(
            X_status, y_status, test_size=0.2, random_state=42, stratify=y_status
        )

        X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
            X_time, y_time, test_size=0.2, random_state=42
        )

        print(f"   Training set: {len(X_train_status)} samples")
        print(f"   Test set: {len(X_test_status)} samples")
        print(f"   Approval rate in training: {y_train_status.mean():.2%}")
        print(f"   Approval rate in test: {y_test_status.mean():.2%}")

        # Train models
        self.status_model, status_acc = train_status_classifier(
            X_train_status, y_train_status, X_test_status, y_test_status
        )

        self.time_model, time_mae = train_processing_time_regressor(
            X_train_time, y_train_time, X_test_time, y_test_time
        )

        print("\nTRAINING COMPLETE\n")
        print(f"Status Classifier Accuracy: {status_acc:.2%}")
        print(f"Processing Time MAE: {time_mae:.2f} days")

        return status_acc, time_mae

    def save_models(self, prefix="visa_model"):
        """Save trained models with absolute path"""
        import os

        # Get absolute path to project root
        current_file = os.path.abspath(__file__)  # Full path to main.py
        src_dir = os.path.dirname(current_file)  # Path to src/ folder
        project_root = os.path.dirname(src_dir)  # Path to project root
        models_dir = os.path.join(project_root, "models")  # Path to models/ folder

        # Create folder if it does not exist
        os.makedirs(models_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"SAVING MODELS")
        print(f"{'=' * 60}")
        print(f"Models directory: {models_dir}")

        # Save with absolute paths
        status_path = os.path.join(models_dir, f"{prefix}_status.pkl")
        time_path = os.path.join(models_dir, f"{prefix}_time.pkl")
        encoders_path = os.path.join(models_dir, f"{prefix}_encoders.pkl")
        features_path = os.path.join(models_dir, f"{prefix}_features.pkl")

        joblib.dump(self.status_model, status_path)
        joblib.dump(self.time_model, time_path)
        joblib.dump(self.label_encoders, encoders_path)
        joblib.dump(self.feature_cols, features_path)

        print(f"✓ Saved: {status_path}")
        print(f"✓ Models saved successfully!")
        print(f"{'=' * 60}\n")

    def load_models(self, prefix="visa_model"):
        """Load trained models with absolute path"""
        import os

        # Get absolute path to project root
        current_file = os.path.abspath(__file__)  # Full path to main.py
        src_dir = os.path.dirname(current_file)  # Path to src/ folder
        project_root = os.path.dirname(src_dir)  # Path to project root
        models_dir = os.path.join(project_root, "models")  # Path to models/ folder

        print(f"\n{'=' * 60}")
        print(f"LOADING MODELS")
        print(f"{'=' * 60}")
        print(f"Models directory: {models_dir}")

        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                f"\n❌ Models directory not found at: {models_dir}\n"
                f"Please run 'python main.py' from the src/ folder to train models first."
            )

        # Build absolute paths
        status_path = os.path.join(models_dir, f"{prefix}_status.pkl")
        time_path = os.path.join(models_dir, f"{prefix}_time.pkl")
        encoders_path = os.path.join(models_dir, f"{prefix}_encoders.pkl")
        features_path = os.path.join(models_dir, f"{prefix}_features.pkl")

        # Check all files exist
        for name, path in [
            ("Status", status_path),
            ("Time", time_path),
            ("Encoders", encoders_path),
            ("Features", features_path),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name} model: {path}")

        # Load the models
        self.status_model = joblib.load(status_path)
        self.time_model = joblib.load(time_path)
        self.label_encoders = joblib.load(encoders_path)
        self.feature_cols = joblib.load(features_path)

        print(f"✓ All models loaded successfully!")
        print(f"{'=' * 60}\n")

    def predict(self, application_data):
        """Predict visa status and processing time for new application"""
        # Convert single dict to DataFrame
        if isinstance(application_data, dict):
            df_app = pd.DataFrame([application_data])
        else:
            df_app = application_data.copy()

        # CHECK HIGH SCRUTINY ROUTE FIRST - AUTOMATIC REJECTION
        applicant_country = df_app.iloc[0]["applicant_country"]
        destination_country = df_app.iloc[0]["destination_country"]

        if (applicant_country, destination_country) in high_scrutiny_routes:
            print(
                f"\n⚠️  HIGH SCRUTINY ROUTE DETECTED: {applicant_country} → {destination_country}"
            )
            print(f"⛔ AUTOMATIC REJECTION APPLIED")

            return {
                "predicted_status": "Rejected",
                "approval_probability": 0.0,
                "rejection_probability": 1.0,
                "estimated_processing_days": 30,  # Standard rejection processing time
                "estimated_processing_weeks": 4.3,
                "rejection_reason": "High Scrutiny Route - Automatic Rejection",
                "high_scrutiny_route": True,
            }

        # Engineer features for non-scrutiny routes
        df_app = engineer_features(df_app)

        # Prepare features
        df_model = df_app.copy()

        # Encode binary columns
        binary_cols = [
            "priority_processing",
            "biometrics_completed",
            "overstay_history",
            "document_completeness",
            "supporting_docs_provided",
            "interview_required",
            "financial_docs_provided",
            "sponsorship_letter",
        ]
        for col in binary_cols:
            df_model[col] = (df_model[col] == "Yes").astype(int)

        # Encode categorical columns
        categorical_cols = [
            "applicant_country",
            "destination_country",
            "application_type",
            "visa_category",
            "season",
            "gender",
            "employment_status",
            "marital_status",
            "processing_office",
            "age_group",
            "travel_experience",
        ]
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                df_model[col + "_encoded"] = (
                    df_model[col]
                    .astype(str)
                    .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                )

        # Get features
        X = df_model[self.feature_cols]

        # Make predictions
        status_pred = self.status_model.predict(X)[0]
        status_proba = self.status_model.predict_proba(X)[0]
        time_pred = max(1, int(self.time_model.predict(X)[0]))

        result = {
            "predicted_status": "Approved" if status_pred == 1 else "Rejected",
            "approval_probability": float(status_proba[1]),
            "rejection_probability": float(status_proba[0]),
            "estimated_processing_days": time_pred,
            "estimated_processing_weeks": round(time_pred / 7, 1),
            "high_scrutiny_route": False,
        }

        return result


# Main execution
if __name__ == "__main__":
    df = load_and_preprocess_data("../data/train.csv")

    df = engineer_features(df)

    df_model, feature_cols, label_encoders = prepare_features_for_modeling(df)
    print(df_model.head())
    print(df_model.info())
    print(
        df_model[
            ["processing_days", "risk_score", "travel_score", "biometrics_completed"]
        ].head()
    )

    print("\nProcessing Time Statistics:")
    print(f"\nMean: {df_model['processing_days'].mean():.2f} days")
    print(f"\nMedian: {df_model['processing_days'].median():.2f} days")
    print(f"\nMin: {df_model['processing_days'].min()} days")
    print(f"\nMax: {df_model['processing_days'].max()} days")
    print(f"\nStd Dev: {df_model['processing_days'].std():.2f} days")

    perform_comprehensive_eda(df, output_dir="eda_outputs")
    # Initialize and train system
    system = VisaPredictionSystem()
    accuracy, mae = system.train(df)
    # Save models
    system.save_models("visa_model")

    print("\nSYSTEM READY FOR PREDICTIONS\n")

    # Example prediction
    print("\nEXAMPLE PREDICTION\n")

    sample_application = {
        "applicant_country": "India",
        "destination_country": "USA",
        "application_type": "In-Person",
        "visa_category": "Tourist",
        "season": "Summer",
        "priority_processing": "No",
        "biometrics_completed": "Yes",
        "countries_visited": 0,
        "schengen_visits": 0,
        "us_visits": 0,
        "uk_visits": 0,
        "overstay_history": "No",
        "previous_rejections": 0,
        "applicant_age": 32,
        "gender": "M",
        "employment_status": "Employed",
        "marital_status": "Married",
        "document_completeness": "Yes",
        "supporting_docs_provided": "Yes",
        "interview_required": "Yes",
        "financial_docs_provided": "Yes",
        "sponsorship_letter": "No",
        "processing_office": "Office A - Metro",
        "submission_month": 6,
        "submission_day_of_week": 2,
        "submission_quarter": 2,
    }

    prediction = system.predict(sample_application)

    print(f"\nApplication Details:")
    print(f"   From: {sample_application['applicant_country']}")
    print(f"   To: {sample_application['destination_country']}")
    print(f"   Type: {sample_application['visa_category']}")

    print(f"\nPredictions:")
    print(f"   Status: {prediction['predicted_status']}")
    print(f"   Approval Probability: {prediction['approval_probability']:.1%}")
    print(
        f"   Estimated Processing Time: {prediction['estimated_processing_days']} days ({prediction['estimated_processing_weeks']} weeks)"
    )
