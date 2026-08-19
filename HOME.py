import streamlit as st
import pandas as pd
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Crop Cultivar Identification",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1.3rem;
        color: #555555;
        text-align: center;
        margin-bottom: 25px;
    }
    .internship-badge {
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 10px 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        margin-bottom: 30px;
        border: 1px solid #c8e6c9;
    }
    .section-header {
        color: #2E7D32;
        border-bottom: 2px solid #2E7D32;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown('<div class="main-title">Machine Learning-Based Identification of Crop Cultivars</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Using Hyperspectral Remote Sensing</div>', unsafe_allow_html=True)

st.markdown("""
<div class="internship-badge">
    🚀 This project was developed during a Summer Internship at the <b>National Remote Sensing Centre (NRSC), ISRO</b>.
</div>
""", unsafe_allow_html=True)

# --- Top Navigation Tabs ---
tab_home, tab_pre, tab_models, tab_comp, tab_contact = st.tabs([
    "🏠 Home", 
    "⚙️ Pre-Processing", 
    "🧠 Models", 
    "📊 Comparison", 
    "📞 Contact Us"
])

# ==========================================
# TAB 1: HOME
# ==========================================
with tab_home:
    col_about, col_method = st.columns(2)

    with col_about:
        st.markdown('<h3 class="section-header">📖 Introduction</h3>', unsafe_allow_html=True)
        st.write("""
        Accurate identification of crop cultivars is essential for precision agriculture, crop monitoring, yield assessment, and sustainable management. 
        
        This interactive dashboard serves as an end-to-end analytical pipeline. It processes hyperspectral reflectance data acquired via an ASD FieldSpec spectroradiometer within the 350–2500 nm spectral range. By bridging raw spectral signatures with advanced predictive modeling, this tool facilitates the rapid, non-destructive discrimination of crop varieties.
        """)

        st.markdown('<h3 class="section-header">📂 Try a Sample Dataset</h3>', unsafe_allow_html=True)
        st.write("Don't have your own data? Download our perfectly formatted sample dataset to test the pipeline.")
        
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
            st.error(f"Sample file not found. Please ensure '{file_path}' is in your directory.")

    with col_method:
        st.markdown('<h3 class="section-header">🔬 Methodology</h3>', unsafe_allow_html=True)
        st.write("The analytical framework follows a structured approach:")
        st.markdown("""
        *   **Spectral Preprocessing:** Removal of noisy spectral bands, resampling to 10 nm intervals, smoothing, and Standard Normal Variate (SNV) transformation.
        *   **Feature Extraction:** Dimensionality reduction and identification of significant wavelengths using Principal Component Analysis (PCA).
        *   **Classification Models:** Implementation of multiple algorithms including Random Forest, SVM, XGBoost, PLS-DA, 1D-CNN, and LSTM.
        *   **Accuracy Assessment:** Evaluation using Overall Accuracy, Kappa Coefficient, Precision, Recall, and F1-score.
        """)

# ==========================================
# TAB 2: PRE-PROCESSING
# ==========================================
with tab_pre:
    st.markdown("### ⚙️ Step 1: Data Pre-Processing")
    st.write("Upload your raw ASD FieldSpec data to begin the pipeline. In this module, you will clean the data, remove atmospheric noise, and resample the spectral bands to prepare for model training.")
    st.page_link("pages/1_PRE-PROCESSING.py", label="Launch Pre-Processing Module", icon="🚀")

# ==========================================
# TAB 3: MODELS
# ==========================================
with tab_models:
    st.markdown("### 🧠 Step 2: Model Training")
    st.write("Utilize your cleaned hyperspectral dataset to train a variety of traditional Machine Learning and Deep Learning classifiers. Tune parameters and extract feature importance.")
    st.page_link("pages/2_MODEL.py", label="Launch Model Training Module", icon="🚀")

# ==========================================
# TAB 4: COMPARISON
# ==========================================
with tab_comp:
    st.markdown("### 📊 Step 3: Model Comparison")
    st.write("Analyze the performance metrics of all your trained models side-by-side. Compare Overall Accuracy, Kappa scores, and visualize confusion matrices to select the best algorithm.")
    st.page_link("pages/3_COMPARISON.py", label="Launch Comparison Module", icon="🚀")

# ==========================================
# TAB 5: CONTACT US
# ==========================================
with tab_contact:
    col_dev, col_guide = st.columns(2)
    
    with col_dev:
        st.markdown('<h3 class="section-header">👨‍💻 Developer</h3>', unsafe_allow_html=True)
        st.write("**Atish** (Enrollment No.: 25AG62R01)")
        st.write("M.Tech Scholar, Land and Water Resources Engineering")
        st.write("Department of Agricultural and Food Engineering")
        st.write("Indian Institute of Technology (IIT), Kharagpur")
        st.write("📧 **Contact:** atishyadav7304@gmail.com")
        st.markdown("[💬 Click here to send feedback or comments](mailto:atishyadav7304@gmail.com?subject=Dashboard%20Feedback%20and%20Comments)")
        
    with col_guide:
        st.markdown('<h3 class="section-header">🎓 Project Supervisors</h3>', unsafe_allow_html=True)
        st.write("**Mr. Laxman Boggarapu** (Project Guidance)")
        st.write("Sci./Eng. 'SE' CAD/ASAG/RSAA")
        st.write("National Remote Sensing Centre (NRSC), ISRO, Hyderabad")
        st.write("---")
        st.write("**Prof. Rajendra Singh** (M.Tech. Supervisor)")
        st.write("Professor (HAG), Dept. AgFE")
        st.write("Indian Institute of Technology (IIT), Kharagpur")

st.markdown("---")
with st.expander("📚 View Project References"):
    st.markdown("""
    1. Goetz, A. F. H., Vane, G., Solomon, J. E., & Rock, B. N. (1985). Imaging Spectrometry for Earth Remote Sensing.
    2. Lu, B., Dao, P. D., Liu, J., He, Y., & Shang, J. (2020). Recent Advances of Hyperspectral Imaging Technology and Applications in Agriculture.
    3. Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedure.
    4. Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra.
    5. Datt, B., et al. (2003). Preprocessing EO-1 Hyperion Hyperspectral Data to Support the Application of Agricultural Indexes.
    """)
