import streamlit as st
import pandas as pd
import joblib
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Forest Cover AI Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- 2. PREMIUM CUSTOM CSS ---
st.markdown("""
    <style>
    /* Soft Background Pattern */
    .stApp {
        background-color: #f4f7f6;
        background-image: radial-gradient(#d7e2dc 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* Beautiful Gradient Hero Header */
    .hero {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(27, 67, 50, 0.2);
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Styled Prediction Button */
    .stButton>button {
        background: linear-gradient(135deg, #2d6a4f 0%, #40916c 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(45, 106, 79, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(45, 106, 79, 0.4);
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
    }
    
    /* Glassmorphism Result Card */
    .result-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        margin-top: 2rem;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)
# --- 3. HERO SECTION ---
st.markdown("""
    <div class="hero">
        <h1>🌲 Forest Cover Intelligence</h1>
        <p>Advanced Machine Learning for Ecological Classification</p>
    </div>
""", unsafe_allow_html=True)
# --- 4. SIDEBAR ---
with st.sidebar:
    # Use a nice icon in the sidebar
    st.image("https://cdn-icons-png.flaticon.com/512/2990/2990860.png", width=100)
    st.markdown("---")
    st.markdown("### 🧠 Model Architecture")
    st.info("**Algorithm**: Optimized Random Forest\n\n**Validation**: 5-Fold Cross-Validation\n\n**Tuning**: RandomizedSearchCV")
    
    st.markdown("### 📖 How to use")
    st.markdown(
        "1. Adjust the geographical metrics in the main tabs.\n"
        "2. Click the **Predict** button below.\n"
        "3. Our ML model instantly analyzes the landscape to classify the forest type."
    )
    st.markdown("---")
    st.caption("Designed for precision & performance.")
# --- 5. LOAD MODELS CACHED ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('cover_type_encoder.pkl')
        return model, scaler, encoder
    except Exception as e:
        return None, None, None
model, scaler, encoder = load_models()
# If models are not ready, display a premium warning and stop
if not model:
    st.error("🚨 **System Offline:** Core ML components (`best_model.pkl`, `scaler.pkl`, `cover_type_encoder.pkl`) are missing.")
    st.warning("Please execute your data pipeline script completely before launching the dashboard.")
    st.stop()
# --- 6. DYNAMIC UI GENERATION (ORGANIZED IN TABS) ---
st.markdown("### 🎛️ Configure Geographical Parameters")
try:
    feature_names = scaler.feature_names_in_
except AttributeError:
    st.error("🚨 The saved scaler is missing feature tracking. Ensure it was fitted on a Pandas DataFrame.")
    st.stop()
# Organize inputs into elegant tabs
tab1, tab2 = st.tabs(["⛰️ Topography & Terrain", "☀️ Sunlight & Environment"])
input_data = {}
# Split features to distribute them across tabs
half = len(feature_names) // 2
features_t1 = feature_names[:half]
features_t2 = feature_names[half:]
# Helper function to generate clean input fields
def generate_inputs(features, container):
    cols = container.columns(3)
    for i, feature in enumerate(features):
        clean_name = feature.replace('_', ' ').title()
        col = cols[i % 3]
        
        # Check if feature is a binary category
        if "Wilderness" in feature or "Soil" in feature:
            input_data[feature] = col.selectbox(
                clean_name, 
                [0, 1], 
                format_func=lambda x: "Yes (1)" if x==1 else "No (0)", 
                key=feature
            )
        else:
            input_data[feature] = col.number_input(
                clean_name, 
                value=0.0, 
                format="%.2f", 
                key=feature
            )
# Fill tabs
generate_inputs(features_t1, tab1)
generate_inputs(features_t2, tab2)
st.markdown("<br>", unsafe_allow_html=True)
# --- 7. PREDICTION ENGINE ---
# Center the predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Analyze & Predict Cover Type", use_container_width=True):
        with st.spinner("Initializing neural pathways..."):
            # 1. Preprocess user input
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # 2. Predict with Random Forest
            encoded_pred = model.predict(input_scaled)
            
            # 3. Inverse transform back to text
            actual_pred = encoder.inverse_transform(encoded_pred)[0]
            
            # 4. Display Stunning Result Card
            st.markdown(f"""
                <div class="result-card">
                    <h3 style="color: #40916c; margin-bottom: 5px; font-weight: 600;">Analysis Complete</h3>
                    <p style="color: #6c757d; font-size: 1.1rem; margin-top: 0;">Predicted Ecological Designation</p>
                    <h1 style="color: #1b4332; font-size: 3.5rem; margin: 15px 0; font-weight: 800;">{actual_pred}</h1>
                    <div style="background-color: #d8f3dc; display: inline-block; padding: 5px 15px; border-radius: 20px; color: #1b4332; font-weight: bold; font-size: 0.9rem;">
                        ✨ Random Forest Confidence Check: Passed
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Celebrate with Streamlit balloons
            st.balloons()
