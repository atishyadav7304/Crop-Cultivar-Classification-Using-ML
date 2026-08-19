import streamlit as st
import pandas as pd
import numpy as np
import io
import streamlit.components.v1 as components
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
    
    /* --- INTENSE BLINKING EFFECT FOR NAVIGATION BUTTONS --- */
    @keyframes hard-blink {
        0%, 100% { 
            opacity: 1; 
            background-color: #e8f5e9; /* Light green background */
            border-color: #2E7D32;
            box-shadow: 0px 4px 10px rgba(46, 125, 50, 0.4);
        }
        50% { 
            opacity: 0.4; 
            background-color: transparent;
            border-color: transparent;
            box-shadow: none;
        }
    }

    /* Target the page links aggressively */
    div[data-testid="stPageLink-NavLink"],
    a[data-testid="stPageLink-NavLink"] {
        animation: hard-blink 1.2s infinite !important;
        border: 2px solid #2E7D32 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin-top: 15px !important;
        display: block !important;
        text-align: center !important;
        font-weight: bold !important;
        transition: all 0.2s ease !important;
    }

    /* Stop blinking and glow solid when the user hovers over it */
    div[data-testid="stPageLink-NavLink"]:hover,
    a[data-testid="stPageLink-NavLink"]:hover {
        animation: none !important;
        opacity: 1 !important;
        background-color: #c8e6c9 !important; /* Darker green on hover */
        transform: scale(1.05) !important;
        box-shadow: 0px 6px 12px rgba(46, 125, 50, 0.6) !important;
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
st.markdown('<h3 class="section-header">👨‍💻 Authors Details and Contact</h3>', unsafe_allow_html=True)

# Create two columns: a larger one for text, a smaller one for the map
col_details, col_map = st.columns([1.5, 1], gap="large")

with col_details:
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

with col_map:
    st.markdown("**📍 Location:** Dept. of AgFE, IIT Kharagpur")
    
    # HTML iframe for embedding Google Maps 
    # Using the exact decimal equivalent of 22°19'03.1"N 87°18'49.5"E (22.317528, 87.313750)
    # This matches your screenshot perfectly.
    map_html = """
    <iframe 
        width="100%" 
        height="320" 
        frameborder="0" 
        style="border:0; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
        scrolling="no" 
        marginheight="0" 
        marginwidth="0" 
        src="https://maps.google.com/maps?q=22.317528,87.313750&hl=en&t=k&z=18&output=embed">
    </iframe>
    """
    
    # Render the HTML map component in Streamlit
    components.html(map_html, height=330)
