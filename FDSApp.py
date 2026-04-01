import os
import joblib
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

# Define Core Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "results", "models", "base_svm.joblib")
FEAT_CSV_PATH = os.path.join(ROOT_DIR, "results", "checkpoints", "selected_feature_names.csv")

# ==========================================
# Helper Functions
# ==========================================
@st.cache_resource
def load_model_and_features():
    """Load the SVM model and the required 74 feature names."""
    try:
        model = joblib.load(MODEL_PATH)
        feat_df = pd.read_csv(FEAT_CSV_PATH)
        required_features = feat_df["feature"].tolist()
        return model, required_features
    except Exception as e:
        return None, None

def predict_fraud(model, df, features, threshold=0.17):
    """Run inference using the SVM model and a custom decision threshold."""
    # Ensure correct column order
    X = df[features].values
    
    # Predict Probabilities
    probas = model.predict_proba(X)[:, 1]
    
    # Apply Threshold
    predictions = (probas >= threshold).astype(int)
    
    # Format Results
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
This application uses a machine-learning **Support Vector Machine (SVM)** to detect illicit transactions (money laundering, ransomware, etc.) on the Bitcoin blockchain.
Upload your transaction feature dataset below to classify transactions.
""")

# Load assets
model, required_features = load_model_and_features()

if model is None:
    st.error(f"🚨 Setup Error: Could not locate the SVM model at `{MODEL_PATH}` or the feature list. Please ensure the pipeline has been run locally first.")
    st.stop()

# ----- Sidebar Configuration -----
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("**Model Selected:** SVM (RBF Kernel)")
    
    st.markdown("---")
    st.subheader("Decision Threshold")
    st.write("Adjust the probability boundary for classifying a transaction as *Illicit*.")
    threshold = st.slider(
        "Threshold (θ)", 
        min_value=0.01, max_value=0.99, value=0.17, step=0.01,
        help="0.17 is the mathematically optimized threshold finding the best balance of F1 and MCC scores."
    )
    
    st.markdown("---")
    st.markdown("**Required Features:** 74")
    with st.expander("View Required Columns", expanded=False):
        st.dataframe(pd.DataFrame(required_features, columns=["Feature Name"]), use_container_width=True)

# ----- Main Area -----

uploaded_file = st.file_uploader("📂 Upload Transaction Data (CSV format)", type=["csv"], help="Must contain the 74 principal and structural features.")

if uploaded_file is not None:
    with st.spinner('Reading Data...'):
        df = pd.read_csv(uploaded_file)
        
    st.success(f"Successfully loaded {len(df):,} transactions.")
    
    # Data Validation
    missing_cols = [f for f in required_features if f not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Uploaded data is missing {len(missing_cols)} required columns.")
        st.write("First 5 missing features:", missing_cols[:5])
        st.stop()
        
    # Execution
    if st.button("🚀 Analyze Transactions", type="primary"):
        with st.spinner('Running SVM Inference...'):
            results = predict_fraud(model, df, required_features, threshold)
            
            illicit_count = int((results["Prediction"] == 1).sum())
            licit_count = len(results) - illicit_count
            illicit_pct = (illicit_count / len(results)) * 100
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Metrics Row
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Analyzed", f"{len(results):,}")
            col2.metric("Detected Licit", f"{licit_count:,}", delta="Safe", delta_color="normal")
            col3.metric("Detected Illicit", f"{illicit_count:,}", delta=f"{illicit_pct:.1f}% of total", delta_color="inverse")
            
            st.markdown("### Transaction Report")
            
            # Style the dataframe for emphasis
            def color_risk(val):
                color = '#ff4b4b' if 'HIGH RISK' in str(val) else '#00cc96'
                return f'color: {color}; font-weight: bold;'
            
            styled_df = results.style.applymap(color_risk, subset=["Risk_Level"])
            st.dataframe(styled_df, use_container_width=True)
            
            # Distribution plot
            st.markdown("### Risk Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.histplot(results, x="Illicit_Probability", hue="Prediction", bins=50, 
                         palette={0: "#00cc96", 1: "#ff4b4b"}, ax=ax)
            ax.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
            ax.set_title("Probability Distribution of Transactions")
            ax.legend()
            st.pyplot(fig)
            
            # Download bounds
            csv_export = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Report (CSV)",
                data=csv_export,
                file_name="fraud_detection_report.csv",
                mime="text/csv",
            )

else:
    st.info("👆 Awaiting CSV upload. You can generate a sample using the `generate_sample.py` script locally if you wish to try the interface.")

