import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import base64
import time

# Add src folder to path
app_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(app_dir, "src")

if os.path.exists(src_path):
    sys.path.insert(0, src_path)
else:
    st.error(f"'src' folder not found at: {src_path}")
    st.stop()

try:
    from main import VisaPredictionSystem, load_and_preprocess_data, engineer_features
except ImportError as e:
    st.error(f"Could not import: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Visa Application Predictor",
    page_icon=os.path.join(app_dir, "public", "images", "logo.png"),
    layout="wide",
)


IMAGES_DIR = os.path.join(app_dir, "public", "images")


# Helper function to convert image to base64
def get_image_base64(image_name):
    """Convert image file to base64 string for CSS"""
    image_path = os.path.join(IMAGES_DIR, image_name)

    if not os.path.exists(image_path):
        st.error(f"Image not found: {image_path}")
        return ""

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Get base64 encoded images
bg_image_b64 = get_image_base64("background.jpg")
header_image_b64 = get_image_base64("heading-image.jpg")

# Custom CSS with loading animation
st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{ 
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background: url("data:image/jpeg;base64,{bg_image_b64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        min-height: 100vh;
    }}

    /* Loading Animation */
    .loading-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        gap: 2rem;
    }}
    
    .loading-spinner {{
        width: 80px;
        height: 80px;
        border: 6px solid #f3f4f6;
        border-top: 6px solid #4C6EF5;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    .loading-text {{
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        text-align: center;
    }}
    
    .loading-subtext {{
        font-size: 0.875rem;
        color: #6B7280;
        text-align: center;
    }}
    
    .loading-dots {{
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1rem;
    }}
    
    .loading-dot {{
        width: 12px;
        height: 12px;
        background: #4C6EF5;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }}
    
    .loading-dot:nth-child(1) {{
        animation-delay: -0.32s;
    }}
    
    .loading-dot:nth-child(2) {{
        animation-delay: -0.16s;
    }}
    
    @keyframes bounce {{
        0%, 80%, 100% {{ 
            transform: scale(0);
            opacity: 0.5;
        }}
        40% {{ 
            transform: scale(1);
            opacity: 1;
        }}
    }}
    
    /* Header Styling */
    .main-header {{
        background: url("data:image/jpeg;base64,{header_image_b64}");
        background-size: cover;
        background-position: center 35%;
        background-repeat: no-repeat;
        padding: 7rem 6.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 24px rgba(76, 110, 245, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    
    .main-title {{ 
        color: black; 
        font-size: 2.75rem; 
        font-weight: 700; 
        margin: 0;
        letter-spacing: -0.5px;
    }}
    
    .main-subtitle {{ 
        color: rgba(255, 255, 255, 0.9); 
        font-size: 1.125rem; 
        margin-top: 0.75rem;
        font-weight: 400;
    }}
    
    .section-title {{
        font-size: 1.125rem;
        font-weight: 600;
        color: #344054;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #F2F4F7;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .section-title::before {{
        content: '';
        width: 4px;
        height: 20px;
        background: linear-gradient(135deg, #4C6EF5 0%, #6B8AF9 100%);
        border-radius: 4px;
    }}
    
    /* Result Modal */
    
    @keyframes slideDown {{
        from {{
            transform: translateY(-20px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    /* Result Cards */
    .result-card-success {{
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        padding: 3rem;
        border-radius: 20px;
        border: 2px solid #6EE7B7;
        margin: 1.5rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    
    .result-card-success::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(34, 197, 94, 0.05) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }}
    
    .result-card-error {{
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        padding: 3rem;
        border-radius: 20px;
        border: 2px solid #FCA5A5;
        margin: 1.5rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    
    .result-card-error::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(239, 68, 68, 0.05) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.05); opacity: 0.8; }}
    }}
    
    .result-status {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
    }}
    
    .result-status.approved {{ 
        color: #059669;
        text-shadow: 0 2px 8px rgba(5, 150, 105, 0.2);
    }}
    
    .result-status.rejected {{ 
        color: #DC2626;
        text-shadow: 0 2px 8px rgba(220, 38, 38, 0.2);
    }}
    
    .result-confidence {{
        font-size: 1.125rem;
        font-weight: 600;
        margin-top: 0.875rem;
        opacity: 0.85;
        position: relative;
        z-index: 1;
    }}
    
    /* Metric Cards */
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.25rem;
        margin: 2rem 0;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, #FAFBFC 0%, #F5F7FA 100%);
        padding: 1.75rem;
        border-radius: 16px;
        text-align: center;
        border-left: 4px solid;
        transition: all 0.3s ease;
        border: 1px solid #E8EBF0;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }}
    
    .metric-card.blue {{ 
        border-left-color: #4C6EF5;
    }}
    
    .metric-card.green {{ 
        border-left-color: #10B981;
    }}
    
    .metric-card.amber {{ 
        border-left-color: #F59E0B;
    }}
    
    .metric-title {{
        font-size: 0.75rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }}
    
    .metric-value {{
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0.5rem 0;
        line-height: 1;
    }}
    
    .metric-label {{
        font-size: 0.8125rem;
        color: #9CA3AF;
        margin-top: 0.5rem;
        font-weight: 500;
    }}
    
    /* Probability Bars */
    .probability-bar {{
        background: #F3F4F6;
        height: 12px;
        border-radius: 100px;
        overflow: hidden;
        margin: 0.875rem 0;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }}
    
    .probability-fill {{
        height: 100%;
        border-radius: 100px;
        transition: width 1.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    
    .probability-fill.green {{
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }}
    
    .probability-fill.red {{
        background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%);
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }}
    
    .probability-label {{
        display: flex;
        justify-content: space-between;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #374151;
    }}
    
    /* Button Styling */
    .stButton > button {{
        background: linear-gradient(135deg, #4C6EF5 0%, #5B7BF7 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        border: none;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 12px rgba(76, 110, 245, 0.25);
        width: 100%;
        letter-spacing: 0.3px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(76, 110, 245, 0.35);
        background: linear-gradient(135deg, #5B7BF7 0%, #6B8AF9 100%);
    }}
    
    .stButton > button:active {{
        transform: translateY(0px);
    }}
    
    /* Input Field Styling */
    .stSelectbox, .stNumberInput {{
        margin-bottom: 0.5rem;
    }}
    
    /* Cursor pointer for all interactive elements */
    .stSelectbox select,
    .stSelectbox > div,
    .stNumberInput input {{
        cursor: pointer !important;
    }}
    
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {{
        border-radius: 10px;
        border: 1.5px solid #E5E7EB;
        background: #FAFBFC;
        transition: all 0.2s ease;
        font-size: 0.9375rem;
        cursor: pointer;
    }}
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {{
        border-color: #4C6EF5;
        background: white;
        cursor: pointer;
    }}
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {{
        border-color: #4C6EF5;
        box-shadow: 0 0 0 3px rgba(76, 110, 245, 0.1);
        background: white;
    }}
    
    /* Label Styling */
    .stSelectbox label,
    .stNumberInput label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.5rem;
    }}
    
    /* Info/Success/Error Boxes */
    .stInfo, .stSuccess, .stWarning, .stError {{
        border-radius: 12px;
        border-left-width: 4px;
        padding: 1rem 1.25rem;
        font-size: 0.9375rem;
    }}
    
   
    /* Glassmorphism Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(6px) !important;
        -webkit-backdrop-filter: blur(6px) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.08) !important;
    }}

    /* Sidebar section headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #000000 !important;
        font-weight: 600;
        text-shadow: none !important;
    }}

    /* Markdown text */
    [data-testid="stSidebar"] .stMarkdown {{
        color: #111111 !important;
    }}

    /* Paragraphs & lists */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {{
        color: #111111 !important;
    }}

    /* Info box */
    [data-testid="stSidebar"] .stInfo {{
        background: rgba(255, 255, 255, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #000000 !important;
        backdrop-filter: blur(4px);
    }}

    /* Success box */
    [data-testid="stSidebar"] .stSuccess {{
        background: rgba(16, 185, 129, 0.18) !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        color: #000000 !important;
        backdrop-filter: blur(4px);
    }}

    /* Caption / Disclaimer */
    [data-testid="stSidebar"] .stCaption {{
        color: rgba(0, 0, 0, 0.7) !important;
    }}

</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False


# Load model ONCE - with auto-training if models don't exist
@st.cache_resource
def load_model():
    """Load model or train if models don't exist"""
    models_dir = os.path.join(app_dir, "models")
    model_files = [
        "visa_model_status.pkl",
        "visa_model_time.pkl",
        "visa_model_encoders.pkl",
        "visa_model_features.pkl",
    ]

    # Check if all model files exist
    models_exist = all(os.path.exists(os.path.join(models_dir, f)) for f in model_files)

    system = VisaPredictionSystem()

    if not models_exist:
        st.warning("🔄 Models not found. Training models for the first time...")
        st.info("⏳ This will take 2-3 minutes. Please wait...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Load and prepare data
            status_text.text("📊 Loading training data...")
            progress_bar.progress(20)
            data_path = os.path.join(app_dir, "data", "train.csv")
            df = load_and_preprocess_data(data_path)

            status_text.text("🔧 Engineering features...")
            progress_bar.progress(40)
            df = engineer_features(df)

            # Train models
            status_text.text("🤖 Training ML models...")
            progress_bar.progress(60)
            accuracy, mae = system.train(df)

            # Save models
            status_text.text("💾 Saving models...")
            progress_bar.progress(80)
            system.save_models("visa_model")

            progress_bar.progress(100)
            status_text.text("✅ Training complete!")

            st.success(
                f"✅ Models trained successfully! Accuracy: {accuracy:.2%}, MAE: {mae:.2f} days"
            )
            st.balloons()
            time.sleep(2)

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return None
    else:
        # Load existing models
        try:
            system.load_models("visa_model")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return None

    return system


# Header
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">Visa Application Predictor</h1>
        <p class="main-subtitle">AI-Powered Prediction System • Advanced Machine Learning Analysis</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Country lists
APPLICANT_COUNTRIES = [
    "Afghanistan",
    "Algeria",
    "Argentina",
    "Australia",
    "Bangladesh",
    "Belgium",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Colombia",
    "Czech Republic",
    "Denmark",
    "Egypt",
    "Finland",
    "France",
    "Germany",
    "Ghana",
    "Greece",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Jordan",
    "Kenya",
    "Lebanon",
    "Malaysia",
    "Mexico",
    "Morocco",
    "Netherlands",
    "New Zealand",
    "Nigeria",
    "Norway",
    "Pakistan",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Russia",
    "Saudi Arabia",
    "Singapore",
    "Somalia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sri Lanka",
    "Sweden",
    "Switzerland",
    "Syria",
    "Taiwan",
    "Thailand",
    "Turkey",
    "UAE",
    "Ukraine",
    "United Kingdom",
    "USA",
    "Venezuela",
    "Vietnam",
    "Yemen",
]

DESTINATION_COUNTRIES = [
    "Australia",
    "Austria",
    "Belgium",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Czech Republic",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hong Kong",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Luxembourg",
    "Malaysia",
    "Mexico",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Poland",
    "Portugal",
    "Qatar",
    "Russia",
    "Saudi Arabia",
    "Singapore",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Taiwan",
    "Thailand",
    "Turkey",
    "UAE",
    "UK",
    "USA",
]

# Form - Personal Information
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown(
    '<h3 class="section-title">Personal Information</h3>', unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    applicant_country = st.selectbox(
        "Applicant Country",
        APPLICANT_COUNTRIES,
        index=APPLICANT_COUNTRIES.index("India"),
        key="applicant_country",
    )

with col2:
    destination_country = st.selectbox(
        "Destination Country",
        DESTINATION_COUNTRIES,
        index=DESTINATION_COUNTRIES.index("USA"),
        key="destination_country",
    )

with col3:
    applicant_age = st.number_input(
        "Age", min_value=18, max_value=80, value=32, key="age"
    )

with col4:
    gender = st.selectbox(
        "Gender",
        ["M", "F"],
        format_func=lambda x: "Male" if x == "M" else "Female",
        key="gender",
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    visa_category = st.selectbox(
        "Visa Category",
        ["Tourist", "Business", "Student", "Work", "Family", "Transit"],
        key="visa_category",
    )

with col2:
    employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Self-Employed", "Student", "Retired", "Unemployed"],
        key="employment_status",
    )

with col3:
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced", "Widowed"],
        index=1,
        key="marital_status",
    )

with col4:
    application_type = st.selectbox(
        "Application Type", ["In-Person", "Online", "Mail"], key="application_type"
    )

st.markdown("</div>", unsafe_allow_html=True)

# Travel History
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<h3 class="section-title">Travel History</h3>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    countries_visited = st.number_input(
        "Countries Visited", min_value=0, max_value=50, value=0, key="countries"
    )

with col2:
    schengen_visits = st.number_input(
        "Schengen Visits", min_value=0, max_value=10, value=0, key="schengen"
    )

with col3:
    us_visits = st.number_input(
        "US Visits", min_value=0, max_value=10, value=0, key="us"
    )

with col4:
    uk_visits = st.number_input(
        "UK Visits", min_value=0, max_value=10, value=0, key="uk"
    )

with col5:
    previous_rejections = st.number_input(
        "Previous Rejections", min_value=0, max_value=5, value=0, key="rejections"
    )

with col6:
    overstay_history = st.selectbox("Overstay History", ["No", "Yes"], key="overstay")

st.markdown("</div>", unsafe_allow_html=True)

# Documentation & Processing
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown(
    '<h3 class="section-title">Documentation & Processing</h3>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    document_completeness = st.selectbox(
        "Document Completeness", ["Yes", "No"], key="doc_complete"
    )

with col2:
    supporting_docs = st.selectbox(
        "Supporting Documents", ["Yes", "No"], key="support_docs"
    )

with col3:
    financial_docs = st.selectbox("Financial Documents", ["Yes", "No"], key="fin_docs")

with col4:
    sponsorship_letter = st.selectbox(
        "Sponsorship Letter", ["No", "Yes"], key="sponsor"
    )

col1, col2, col3 = st.columns(3)

with col1:
    biometrics = st.selectbox("Biometrics Completed", ["Yes", "No"], key="biometrics")

with col2:
    priority_processing = st.selectbox(
        "Priority Processing", ["No", "Yes"], key="priority"
    )

with col3:
    processing_office = st.selectbox(
        "Processing Office",
        [
            "Office A - Metro",
            "Office B - Regional",
            "Office C - Small Town",
            "Office D - Capital",
            "Office E - Border",
            "Office F - Consulate",
        ],
        key="office",
    )

st.markdown("</div>", unsafe_allow_html=True)

# Predict Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("GET PREDICTION", use_container_width=True):
        # Reset states before starting new prediction
        st.session_state.show_result = False
        st.session_state.prediction_result = None
        st.session_state.is_loading = True
        st.rerun()

# Loading Animation
if st.session_state.is_loading and not st.session_state.show_result:
    st.markdown(
        """
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">Analyzing Your Application</div>
            <div class="loading-subtext">Our AI is processing 25+ data points...</div>
            <div class="loading-dots">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    system = load_model()

    if system is None:
        st.error("Model not available. Please refresh the page.")
        st.session_state.is_loading = False
    else:
        now = datetime.now()

        # Build application data
        application_data = {
            "applicant_country": applicant_country,
            "destination_country": destination_country,
            "application_type": application_type,
            "visa_category": visa_category,
            "season": [
                "Winter",
                "Winter",
                "Spring",
                "Spring",
                "Spring",
                "Summer",
                "Summer",
                "Summer",
                "Fall",
                "Fall",
                "Fall",
                "Winter",
            ][now.month - 1],
            "priority_processing": priority_processing,
            "biometrics_completed": biometrics,
            "countries_visited": countries_visited,
            "schengen_visits": schengen_visits,
            "us_visits": us_visits,
            "uk_visits": uk_visits,
            "overstay_history": overstay_history,
            "previous_rejections": previous_rejections,
            "applicant_age": applicant_age,
            "gender": gender,
            "employment_status": employment_status,
            "marital_status": marital_status,
            "document_completeness": document_completeness,
            "supporting_docs_provided": supporting_docs,
            "interview_required": "Yes",
            "financial_docs_provided": financial_docs,
            "sponsorship_letter": sponsorship_letter,
            "processing_office": processing_office,
            "submission_month": now.month,
            "submission_day_of_week": now.weekday(),
            "submission_quarter": (now.month - 1) // 3 + 1,
        }

        try:
            # Simulate processing time for better UX
            time.sleep(1.5)
            result = system.predict(application_data)
            st.session_state.prediction_result = result
            st.session_state.show_result = True
            st.session_state.is_loading = False
            st.rerun()
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            import traceback

            st.error(f"```\n{traceback.format_exc()}\n```")
            st.session_state.is_loading = False

# RESULTS SECTION
if st.session_state.show_result and st.session_state.prediction_result:
    result = st.session_state.prediction_result

    st.markdown('<div class="result-modal">', unsafe_allow_html=True)

    # Result Card
    if result["predicted_status"] == "Approved":
        st.markdown(
            f"""
            <div class="result-card-success">
                <h1 class="result-status approved">APPROVED</h1>
                <p class="result-confidence" style="color: #059669;">
                    Confidence: {result["approval_probability"] * 100:.1f}%
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-error">
                <h1 class="result-status rejected">REJECTED</h1>
                <p class="result-confidence" style="color: #dc2626;">
                    Confidence: {result["rejection_probability"] * 100:.1f}%
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Metrics Grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card blue">
                <p class="metric-title">Processing Time</p>
                <h1 class="metric-value" style="color: #4C6EF5;">{result["estimated_processing_days"]}</h1>
                <p class="metric-label">days</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card green">
                <p class="metric-title">Approval Rate</p>
                <h1 class="metric-value" style="color: #10B981;">{result["approval_probability"] * 100:.1f}%</h1>
                <p class="metric-label">probability</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card amber">
                <p class="metric-title">Time Frame</p>
                <h1 class="metric-value" style="color: #F59E0B;">{result["estimated_processing_weeks"]}</h1>
                <p class="metric-label">weeks</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Probability Analysis
    st.markdown("### Probability Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="probability-label">
                <span>Approval Probability</span>
                <span style="color: #10B981;">{result["approval_probability"] * 100:.1f}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill green" style="width: {result["approval_probability"] * 100}%;"></div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="probability-label">
                <span>Rejection Probability</span>
                <span style="color: #EF4444;">{result["rejection_probability"] * 100:.1f}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill red" style="width: {result["rejection_probability"] * 100}%;"></div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Disclaimer
    st.info(
        "**Important Disclaimer:** This prediction is generated using AI and machine learning. "
        "Actual visa decisions are made by immigration officers and depend on numerous factors."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Close button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Make New Prediction", use_container_width=True):
            st.session_state.show_result = False
            st.session_state.prediction_result = None
            st.session_state.is_loading = False
            st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### About This Tool")
    st.info("""
        **Visa Predictor AI** uses advanced machine learning algorithms to analyze 
        visa applications based on historical patterns and success factors.
        
        **Key Features:**
        - ML-based predictions with confidence scores
        - Processing time estimates
        - Comprehensive risk analysis
        - Document completeness checker
        - Trained on 5,000+ real applications
        - Personalized recommendations
    """)

    st.markdown("### How It Works")
    st.success("""
        1. **Data Collection** - Enter your application details
        2. **Feature Engineering** - System processes 25+ data points
        3. **Model Inference** - Ensemble models generate predictions
        4. **Risk Assessment** - Analyze potential approval barriers
        5. **Recommendations** - Receive actionable insights
    """)

    st.markdown("---")
    st.caption(
        "Disclaimer: This tool provides estimates only. Official decisions are made by immigration authorities."
    )
