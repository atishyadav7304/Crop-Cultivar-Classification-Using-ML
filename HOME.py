import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Crop Cultivar Identification",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling & Blinking Buttons ---
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        color: #2E7D32;
        border-bottom: 2px solid #2E7D32;
        padding-bottom: 5px;
        margin-top: 30px;
    }
    .footer {
        text-align: center;
        font-size: 0.95rem;
        color: #555555;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e0e0e0;
        background-color: #f9f9f9;
        padding-bottom: 20px;
        border-radius: 10px;
    }
    
    /* --- NEW: Glowing/Blinking Animation for Navigation Buttons --- */
    @keyframes glowing-pulse {
        0% { box-shadow: 0 0 5px rgba(46, 125, 50, 0.2); transform: scale(1); }
        50% { box-shadow: 0 0 20px rgba(46, 125, 50, 0.9); transform: scale(1.03); }
        100% { box-shadow: 0 0 5px rgba(46, 125, 50, 0.2); transform: scale(1); }
    }

    /* Target ONLY the page links in the main body area */
    .main a[data-testid="stPageLink-NavLink"] {
        animation: glowing-pulse 1.5s infinite !important;
        background-color: #ffffff !important;
        border: 2px solid #2E7D32 !important;
        border-radius: 8px !important;
        margin-top: 10px !important;
        transition: all 0.2s ease !important;
    }

    /* Lock the button in a highlighted state when the user hovers over it */
    .main a[data-testid="stPageLink-NavLink"]:hover {
        animation: none !important;
        transform: scale(1.05) !important;
        background-color: #e8f5e9 !important;
        box-shadow: 0 0 15px rgba(46, 125, 50, 1) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown('<div class="main-title">Machine Learning-Based Identification of Crop Cultivars</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Using Hyperspectral Remote Sensing</div>', unsafe_allow_html=True)

# --- About & Methodology Section ---
col_about, col_method = st.columns(2)

with col_about:
    st.markdown('<h3 class="section-header">📖 About This Dashboard</h3>', unsafe_allow_html=True)
    st.write("""
    Accurate identification of crop cultivars is essential for precision agriculture, crop monitoring, yield assessment, and sustainable management. 
    
    This interactive dashboard serves as an end-to-end analytical pipeline. It processes hyperspectral reflectance data acquired via an ASD FieldSpec spectroradiometer within the 350–2500 nm spectral range. By bridging raw spectral signatures with advanced predictive modeling, this tool facilitates the rapid, non-destructive discrimination of crop varieties.
    """)

with col_method:
    st.markdown('<h3 class="section-header">🔬 Methodology</h3>', unsafe_allow_html=True)
    st.write("The analytical framework follows a structured approach:")
    st.markdown("""
    *   **Spectral Preprocessing:** Removal of noisy spectral bands, resampling to 10 nm intervals, smoothing, and Standard Normal Variate (SNV) transformation to correct scattering.
    *   **Feature Extraction:** Dimensionality reduction and identification of significant wavelengths using Principal Component Analysis (PCA).
    *   **Classification Models:** Implementation of 7 algorithms, including Logistic Regression, Random Forest, XGBoost, Support Vector Machine (SVM), PLS-DA, 1D-CNN, and LSTM.
    *   **Accuracy Assessment:** Evaluation using Overall Accuracy, Kappa Coefficient, Precision, Recall, and F1-score.
    """)

st.markdown("---")

# --- Navigation Buttons ---
st.write("### 🚀 Dashboard Navigation")
st.write("Select a module to begin processing your data:")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Step 1: Pre-Processing**")
    st.write("Upload raw ASD FieldSpec data, remove noise, and resample to 10nm intervals.")
    st.page_link("pages/1_PRE-PROCESSING.py", label="Go to Pre-Processing", icon="⚙️")

with col2:
    st.success("**Step 2: Model Training**")
    st.write("Train traditional ML and Deep Learning algorithms on your split datasets.")
    st.page_link("pages/2_MODEL.py", label="Go to Models", icon="🧠")

with col3:
    st.warning("**Step 3: Comparison**")
    st.write("Analyze and compare evaluation metrics across all trained models.")
    st.page_link("pages/3_COMPARISON.py", label="Go to Comparison", icon="📊")

st.markdown("---")

# --- Sample Data Download Section ---
st.write("### 📂 Don't have data? Try a sample!")
st.write("Download a sample hyperspectral dataset formatted specifically for this dashboard. You can use this file in **Step 1: Pre-Processing**.")

file_path = "Ujjain_All_Crops_Cultivar_Demo_Dashboard.xlsx" 

try:
    with open(file_path, "rb") as file:
        st.download_button(
            label="📥 Download Sample Hyperspectral Data (.xlsx)",
            data=file,
            file_name="Sample_Hyperspectral_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
except FileNotFoundError:
    st.error(f"Sample file not found. Please ensure '{file_path}' is uploaded to the main GitHub repository.")

# --- References Section ---
with st.expander("📚 View Project References"):
    st.markdown("""
    1. Goetz, A. F. H., Vane, G., Solomon, J. E., & Rock, B. N. (1985). Imaging Spectrometry for Earth Remote Sensing.
    2. Lu, B., Dao, P. D., Liu, J., He, Y., & Shang, J. (2020). Recent Advances of Hyperspectral Imaging Technology and Applications in Agriculture.
    3. Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedure.
    4. Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra.
    5. Datt, B., et al. (2003). Preprocessing EO-1 Hyperion Hyperspectral Data to Support the Application of Agricultural Indexes.
    """)

# --- Developer & Contact Details (Footer) ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
    **Developed by:** Atish (Enrollment No.: 25AG62R01) <br>
    M.Tech Scholar, Land and Water Resource Engineering <br>
    <br>
    **M.Tech. Supervisor:** Prof. Rajendra Singh <br>
    Professor (HAG), Dept. AgFE, Indian Institute of Technology, Kharagpur<br>
    <br>
    **Project Guidance:** Mr. Laxman Boggarapu <br>
    Sci./Eng. 'SE' CAD/ASAG/RSAA, National Remote Sensing Centre (NRSC), ISRO, Hyderabad <br>
    <br>
    **Institutions:** <br>
    Department of Agricultural and Food Engineering, Indian Institute of Technology Kharagpur <br>
    National Remote Sensing Centre (NRSC), ISRO, Hyderabad <br>
    <br>
    📧 **Contact:** atishyadav7304@gmail.com <br>
    💬 **Feedback:** We value your input! <a href="mailto:atishyadav7304@gmail.com?subject=Dashboard%20Feedback%20and%20Comments">Click here to send feedback or comments</a>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
