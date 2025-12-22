import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def perform_comprehensive_eda(df, output_dir="eda_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Dataset Overview
    print("\n1.DATASET OVERVIEW")
    print(f"Total Records: {len(df)}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Date Range: {df['submission_date'].min()} to {df['submission_date'].max()}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Target Variable Distribution
    print("\n2.TARGET VARIABLE ANALYSIS")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Application Status Distribution
    status_counts = df["application_status"].value_counts()
    axes[0].pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
    )
    axes[0].set_title("Application Status Distribution", fontsize=14, fontweight="bold")

    # Processing Days Distribution
    axes[1].hist(df["processing_days"], bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Processing Days")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Processing Time Distribution", fontsize=14, fontweight="bold")
    axes[1].axvline(
        df["processing_days"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['processing_days'].mean():.1f} days",
    )
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/01_target_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Approval Rate: {(df['application_status'] == 'Approved').mean():.2%}")
    print(f"Average Processing Time: {df['processing_days'].mean():.1f} days")
    print(f"Median Processing Time: {df['processing_days'].median():.1f} days")

    # Visa Category Analysis
    print("\n3.VISA CATEGORY ANALYSIS")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Visa category distribution
    visa_counts = df["visa_category"].value_counts()
    axes[0, 0].barh(visa_counts.index, visa_counts.values, color="skyblue")
    axes[0, 0].set_xlabel("Count")
    axes[0, 0].set_title("Application Count by Visa Category", fontweight="bold")

    # Approval rate by visa category
    approval_by_visa = (
        df.groupby("visa_category")["application_status"]
        .apply(lambda x: (x == "Approved").mean() * 100)
        .sort_values()
    )
    axes[0, 1].barh(approval_by_visa.index, approval_by_visa.values, color="lightgreen")
    axes[0, 1].set_xlabel("Approval Rate (%)")
    axes[0, 1].set_title("Approval Rate by Visa Category", fontweight="bold")

    # Processing time by visa category
    df.boxplot(column="processing_days", by="visa_category", ax=axes[1, 0])
    axes[1, 0].set_title("Processing Time by Visa Category", fontweight="bold")
    axes[1, 0].set_xlabel("Visa Category")
    axes[1, 0].set_ylabel("Processing Days")
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45, ha="right")

    # Visa category vs season heatmap
    season_visa_pivot = df.pivot_table(
        values="processing_days",
        index="visa_category",
        columns="season",
        aggfunc="mean",
    )
    sns.heatmap(season_visa_pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1, 1])
    axes[1, 1].set_title(
        "Avg Processing Days: Visa Category vs Season", fontweight="bold"
    )

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/02_visa_category_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Geographic Analysis

    print("\n4.GEOGRAPHIC ANALYSIS")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top applicant countries
    top_applicants = df["applicant_country"].value_counts().head(10)
    axes[0, 0].barh(top_applicants.index, top_applicants.values, color="coral")
    axes[0, 0].set_xlabel("Applications")
    axes[0, 0].set_title("Top 10 Applicant Countries", fontweight="bold")
    axes[0, 0].invert_yaxis()

    # Top destination countries
    top_destinations = df["destination_country"].value_counts().head(10)
    axes[0, 1].barh(top_destinations.index, top_destinations.values, color="lightblue")
    axes[0, 1].set_xlabel("Applications")
    axes[0, 1].set_title("Top 10 Destination Countries", fontweight="bold")
    axes[0, 1].invert_yaxis()

    # Approval rate by top applicant countries
    top_country_approval = (
        df[df["applicant_country"].isin(top_applicants.index)]
        .groupby("applicant_country")["application_status"]
        .apply(lambda x: (x == "Approved").mean() * 100)
        .sort_values()
    )
    axes[1, 0].barh(
        top_country_approval.index, top_country_approval.values, color="lightgreen"
    )
    axes[1, 0].set_xlabel("Approval Rate (%)")
    axes[1, 0].set_title("Approval Rate by Top Applicant Countries", fontweight="bold")

    # Processing time by destination
    top_dest_processing = (
        df[df["destination_country"].isin(top_destinations.index)]
        .groupby("destination_country")["processing_days"]
        .mean()
        .sort_values()
    )
    axes[1, 1].barh(
        top_dest_processing.index, top_dest_processing.values, color="orange"
    )
    axes[1, 1].set_xlabel("Avg Processing Days")
    axes[1, 1].set_title("Avg Processing Time by Destination", fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/03_geographic_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Correlation Analysis
    print("\n5.CORRELATION ANALYSIS")

    # Numerical features correlation
    numerical_cols = [
        "applicant_age",
        "countries_visited",
        "schengen_visits",
        "us_visits",
        "uk_visits",
        "previous_rejections",
        "processing_days",
        "travel_score",
        "doc_quality_score",
        "risk_score",
    ]

    correlation_matrix = df[numerical_cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Full correlation heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=axes[0],
        square=True,
    )
    axes[0].set_title("Feature Correlation Matrix", fontweight="bold", fontsize=14)

    # Processing time correlations
    processing_corr = correlation_matrix["processing_days"].sort_values(
        ascending=False
    )[1:]
    colors = ["green" if x > 0 else "red" for x in processing_corr.values]
    axes[1].barh(processing_corr.index, processing_corr.values, color=colors, alpha=0.7)
    axes[1].set_xlabel("Correlation with Processing Days")
    axes[1].set_title(
        "Feature Correlation with Processing Time", fontweight="bold", fontsize=14
    )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/04_correlation_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("\nTop 5 Features Correlated with Processing Time:")
    print(processing_corr.head())

    # Travel History Impact
    print("\n6.TRAVEL HISTORY IMPACT ANALYSIS")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Travel experience vs approval
    travel_approval = df.groupby("travel_experience")["application_status"].apply(
        lambda x: (x == "Approved").mean() * 100
    )
    axes[0, 0].bar(
        travel_approval.index, travel_approval.values, color="teal", alpha=0.7
    )
    axes[0, 0].set_ylabel("Approval Rate (%)")
    axes[0, 0].set_title("Approval Rate by Travel Experience", fontweight="bold")
    axes[0, 0].set_xticklabels(travel_approval.index, rotation=45)

    # Previous rejections impact
    rejection_impact = df.groupby("previous_rejections")["application_status"].apply(
        lambda x: (x == "Approved").mean() * 100
    )
    axes[0, 1].plot(
        rejection_impact.index,
        rejection_impact.values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="red",
    )
    axes[0, 1].set_xlabel("Previous Rejections")
    axes[0, 1].set_ylabel("Approval Rate (%)")
    axes[0, 1].set_title("Impact of Previous Rejections on Approval", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Overstay history impact
    overstay_data = df.groupby("overstay_history")["application_status"].apply(
        lambda x: (x == "Approved").mean() * 100
    )
    axes[1, 0].bar(
        overstay_data.index, overstay_data.values, color=["green", "red"], alpha=0.7
    )
    axes[1, 0].set_ylabel("Approval Rate (%)")
    axes[1, 0].set_title("Overstay History Impact on Approval", fontweight="bold")

    # Travel score distribution by status
    df.boxplot(column="travel_score", by="application_status", ax=axes[1, 1])
    axes[1, 1].set_title("Travel Score Distribution by Status", fontweight="bold")
    axes[1, 1].set_xlabel("Application Status")
    axes[1, 1].set_ylabel("Travel Score")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/05_travel_history_impact.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Document Quality Analysis
    print("\n7.DOCUMENT QUALITY ANALYSIS")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Document completeness impact
    doc_complete_approval = df.groupby("document_completeness")[
        "application_status"
    ].apply(lambda x: (x == "Approved").mean() * 100)
    axes[0, 0].bar(
        doc_complete_approval.index,
        doc_complete_approval.values,
        color=["red", "green"],
        alpha=0.7,
    )
    axes[0, 0].set_ylabel("Approval Rate (%)")
    axes[0, 0].set_title("Document Completeness Impact", fontweight="bold")

    # Financial docs impact
    financial_approval = df.groupby("financial_docs_provided")[
        "application_status"
    ].apply(lambda x: (x == "Approved").mean() * 100)
    axes[0, 1].bar(
        financial_approval.index,
        financial_approval.values,
        color=["red", "green"],
        alpha=0.7,
    )
    axes[0, 1].set_ylabel("Approval Rate (%)")
    axes[0, 1].set_title("Financial Documentation Impact", fontweight="bold")

    # Doc quality score vs processing time
    axes[1, 0].scatter(
        df["doc_quality_score"],
        df["processing_days"],
        alpha=0.3,
        c=df["application_status"].map({"Approved": "green", "Rejected": "red"}),
    )
    axes[1, 0].set_xlabel("Document Quality Score")
    axes[1, 0].set_ylabel("Processing Days")
    axes[1, 0].set_title("Doc Quality vs Processing Time", fontweight="bold")

    # Risk score distribution by status
    df.boxplot(column="risk_score", by="application_status", ax=axes[1, 1])
    axes[1, 1].set_title("Risk Score Distribution by Status", fontweight="bold")
    axes[1, 1].set_xlabel("Application Status")
    axes[1, 1].set_ylabel("Risk Score")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/06_document_quality_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    #  Seasonal Patterns
    print("\n8.SEASONAL PATTERNS ANALYSIS")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Applications by season
    season_counts = df["season"].value_counts()
    axes[0, 0].pie(
        season_counts.values,
        labels=season_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0, 0].set_title("Applications by Season", fontweight="bold")

    # Processing time by season
    season_processing = df.groupby("season")["processing_days"].mean().sort_values()
    axes[0, 1].barh(
        season_processing.index, season_processing.values, color="orange", alpha=0.7
    )
    axes[0, 1].set_xlabel("Avg Processing Days")
    axes[0, 1].set_title("Avg Processing Time by Season", fontweight="bold")

    # Monthly trend
    monthly_apps = df.groupby("submission_month").size()
    axes[1, 0].plot(
        monthly_apps.index, monthly_apps.values, marker="o", linewidth=2, markersize=8
    )
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Applications")
    axes[1, 0].set_title("Monthly Application Trend", fontweight="bold")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].grid(True, alpha=0.3)

    # Seasonal approval rates
    season_approval = df.groupby("season")["application_status"].apply(
        lambda x: (x == "Approved").mean() * 100
    )
    axes[1, 1].bar(
        season_approval.index, season_approval.values, color="lightgreen", alpha=0.7
    )
    axes[1, 1].set_ylabel("Approval Rate (%)")
    axes[1, 1].set_title("Approval Rate by Season", fontweight="bold")
    axes[1, 1].set_xticklabels(season_approval.index, rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_seasonal_patterns.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Demographics Analysis
    print("\n9.DEMOGRAPHICS ANALYSIS")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Age distribution
    axes[0, 0].hist(
        df["applicant_age"], bins=30, edgecolor="black", alpha=0.7, color="skyblue"
    )
    axes[0, 0].set_xlabel("Age")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Age Distribution", fontweight="bold")
    axes[0, 0].axvline(
        df["applicant_age"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['applicant_age'].mean():.1f}",
    )
    axes[0, 0].legend()

    # Age group approval rates
    age_group_approval = df.groupby("age_group")["application_status"].apply(
        lambda x: (x == "Approved").mean() * 100
    )
    axes[0, 1].bar(
        age_group_approval.index, age_group_approval.values, color="coral", alpha=0.7
    )
    axes[0, 1].set_ylabel("Approval Rate (%)")
    axes[0, 1].set_title("Approval Rate by Age Group", fontweight="bold")
    axes[0, 1].set_xticklabels(age_group_approval.index, rotation=45)

    # Employment status impact
    employment_approval = (
        df.groupby("employment_status")["application_status"]
        .apply(lambda x: (x == "Approved").mean() * 100)
        .sort_values()
    )
    axes[1, 0].barh(
        employment_approval.index,
        employment_approval.values,
        color="lightgreen",
        alpha=0.7,
    )
    axes[1, 0].set_xlabel("Approval Rate (%)")
    axes[1, 0].set_title("Approval Rate by Employment Status", fontweight="bold")

    # Marital status impact
    marital_approval = df.groupby("marital_status")["application_status"].apply(
        lambda x: (x == "Approved").mean() * 100
    )
    axes[1, 1].bar(
        marital_approval.index, marital_approval.values, color="purple", alpha=0.7
    )
    axes[1, 1].set_ylabel("Approval Rate (%)")
    axes[1, 1].set_title("Approval Rate by Marital Status", fontweight="bold")
    axes[1, 1].set_xticklabels(marital_approval.index, rotation=45)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/08_demographics_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Key Insights Summary
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FROM EDA")
    print("=" * 80)

    print("\nTop Correlations with Processing Time:")
    print(processing_corr.head())

    print("\nHighest Approval Rates:")
    print(
        f"  - Travel Experience: {travel_approval.idxmax()} ({travel_approval.max():.1f}%)"
    )
    print(
        f"  - Visa Category: {approval_by_visa.idxmax()} ({approval_by_visa.max():.1f}%)"
    )
    print(f"  - Season: {season_approval.idxmax()} ({season_approval.max():.1f}%)")

    print("\nCritical Risk Factors:")
    print(f"  - Overstay History reduces approval by {100 - overstay_data['Yes']:.1f}%")
    print(
        f"  - Incomplete Documents reduce approval by {100 - doc_complete_approval['No']:.1f}%"
    )
    print(
        f"  - Each Previous Rejection reduces approval by ~{rejection_impact[0] - rejection_impact[1]:.1f}%"
    )

    print(f"\nEDA Complete! Visualizations saved to '{output_dir}/' directory")
    print("=" * 80)
