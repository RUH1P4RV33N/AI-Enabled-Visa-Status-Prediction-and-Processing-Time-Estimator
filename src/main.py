import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


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
    print("\nAdding features...")

    # Travel history score
    df["travel_score"] = (
        df["countries_visited"] * 2
        + df["schengen_visits"] * 4
        + df["us_visits"] * 5
        + df["uk_visits"] * 3
    ) / 95

    # Documentation quality score
    df["doc_quality_score"] = (
        (df["document_completeness"] == "Yes").astype(int) * 0.4
        + (df["supporting_docs_provided"] == "Yes").astype(int) * 0.3
        + (df["financial_docs_provided"] == "Yes").astype(int) * 0.3
    )

    # Risk factors
    df["risk_score"] = (
        (df["overstay_history"] == "Yes").astype(int) * 0.3
        + df["previous_rejections"] * 0.15
        + (df["document_completeness"] == "No").astype(int) * 0.25
        + (df["supporting_docs_provided"] == "No").astype(int) * 0.15
        + (df["financial_docs_provided"] == "No").astype(int) * 0.15
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

    # High security screening routes
    high_scrutiny_routes = [
        ("India", "USA"),
        ("China", "USA"),
        ("Pakistan", "USA"),
        ("Nigeria", "USA"),
        ("Nigeria", "UK"),
        ("Bangladesh", "USA"),
        ("Iran", "USA"),
        ("Syria", "Germany"),
        ("Afghanistan", "USA"),
        ("Iraq", "UK"),
        ("Somalia", "USA"),
        ("Yemen", "UK"),
        ("India", "UK"),
        ("Pakistan", "UK"),
        ("Egypt", "USA"),
        ("Algeria", "France"),
        ("Pakistan", "France"),
        ("Pakistan", "Germany"),
        ("Afghanistan", "Germany"),
        ("Syria", "UK"),
        ("Iraq", "France"),
        ("Somalia", "UK"),
    ]
    df["high_scrutiny_route"] = df.apply(
        lambda x: 1
        if (x["applicant_country"], x["destination_country"]) in high_scrutiny_routes
        else 0,
        axis=1,
    )

    print(f"\nCreated {len(df.columns)} total features\n")
    print(df.columns.to_list())
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

    return df_model, label_encoders


# Main execution
if __name__ == "__main__":
    df = load_and_preprocess_data("../data/train.csv")

    df = engineer_features(df)

    df_model, label_encoders = prepare_features_for_modeling(df)
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
