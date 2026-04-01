import os
import joblib
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Configuration & Setup
# ==========================================
st.set_page_config(
    page_title="Elliptic Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit menu and footer for a cleaner look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define Core Paths & Helper Functions
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_files_extracted():
    """Extract results.zip if the directory is missing."""
    zip_name = "results.zip"
    target_dir = "results"
    zip_path = os.path.join(ROOT_DIR, zip_name)
    dir_path = os.path.join(ROOT_DIR, target_dir)
    
    if not os.path.exists(dir_path) and os.path.exists(zip_path):
        with st.spinner(f"Extracting {zip_name}..."):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(ROOT_DIR)
                st.toast(f"Successfully extracted {zip_name}", icon="📦")
            except Exception as e:
                st.error(f"Error extracting {zip_name}: {e}")

def find_file(filename, subfolders):
    """Search for a file in root or specific subfolders."""
    # 1. Check Root
    root_path = os.path.join(ROOT_DIR, filename)
    if os.path.exists(root_path):
        return root_path
    
    # 2. Check Subfolders
    for folder in subfolders:
        layered_path = os.path.join(ROOT_DIR, *folder.split('/'), filename)
        if os.path.exists(layered_path):
            return layered_path
            
    return None

# Perform pre-flight extraction if needed
ensure_files_extracted()

# Resolve Final Paths (Supporting root-level uploads like on your GitHub)
MODEL_PATH = find_file("base_svm.joblib", ["results/models", "result/models"])
FEAT_CSV_PATH = find_file("selected_feature_names.csv", ["results/checkpoints", "result/checkpoints"])

# ==========================================
# Helper Functions
# ==========================================
@st.cache_resource
def load_model_and_features():
    """Load the SVM model and the required 74 feature names."""
    if not MODEL_PATH or not FEAT_CSV_PATH:
        return None, None
        
    try:
        model = joblib.load(MODEL_PATH)
        feat_df = pd.read_csv(FEAT_CSV_PATH)
        required_features = feat_df["feature"].tolist()
        return model, required_features
    except Exception as e:
        st.sidebar.error(f"Error loading assets: {e}")
        return None, None

def predict_fraud(model, df, features, threshold=0.17):
    """Run inference using the SVM model and a custom decision threshold."""
    # Ensure correct column order
    X = df[features].values
    probas = model.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)
    
    results_df = df.copy()
    if "txId" in results_df.columns:
        results_df = results_df[["txId"]]
    else:
        results_df = pd.DataFrame({"Transaction_Index": range(len(results_df))})
        
    results_df["Illicit_Probability"] = np.round(probas, 4)
    results_df["Prediction"] = predictions
    results_df["Risk_Level"] = results_df["Prediction"].map({1: "🚨 HIGH RISK (Illicit)", 0: "✅ LOW RISK (Licit)"})
    
    return results_df

# ==========================================
# Application UI
# ==========================================
st.title("🛡️ Bitcoin Fraud Detection System")
st.markdown("""
This application uses a machine-learning **Support Vector Machine (SVM)** to detect illicit transactions on the Bitcoin blockchain.
**Developed by: Md Ahasan Kabir**
""")

# Load assets
model, required_features = load_model_and_features()

if model is None:
    st.error("🚨 **Setup Error: Missing Required Assets**")
    st.markdown(f"""
    Could not locate the SVM model or the feature list. 
    **Checked Root Folder for:** `base_svm.joblib` and `selected_feature_names.csv`.
    """)
    st.stop()

# ----- Sidebar Configuration -----
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # System Health Check
    st.subheader("System Health")
    model_status = "✅ Found" if MODEL_PATH else "❌ Missing"
    feat_status = "✅ Found" if FEAT_CSV_PATH else "❌ Missing"
    
    st.caption(f"Model File: {model_status}")
    st.caption(f"Feature List: {feat_status}")
    
    if MODEL_PATH:
        with st.expander("Show Location", expanded=False):
            st.code(f"Path: {MODEL_PATH}")

    st.markdown("---")
    st.markdown("**Model Selected:** SVM (RBF Kernel)")
    
    threshold = st.slider("Threshold (θ)", min_value=0.01, max_value=0.99, value=0.17, step=0.01)

# ----- Main Area -----
uploaded_file = st.file_uploader("📂 Upload Transaction Data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Successfully loaded {len(df):,} transactions.")
    
    missing_cols = [f for f in required_features if f not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing {len(missing_cols)} columns.")
        st.stop()
        
    if st.button("🚀 Analyze Transactions", type="primary"):
        with st.spinner('Running SVM Inference...'):
            results = predict_fraud(model, df, required_features, threshold)
            
            illicit_count = int((results["Prediction"] == 1).sum())
            st.metric("Total Analyzed", f"{len(results):,}")
            st.metric("Detected Illicit", f"{illicit_count:,}")
            
            st.dataframe(results, use_container_width=True)
