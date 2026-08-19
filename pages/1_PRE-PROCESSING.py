import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# --- Helper Function for Plotting Spectral Data ---
def plot_spectra(data, title):
    """
    Groups data by the first column (cultivar), takes 1 sample per cultivar, 
    and plots Reflectance vs Wavelength mimicking the academic scientific layout.
    """
    sample_df = data.groupby(data.columns[0]).first()
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
    sample_df_t = sample_df.T
    sample_df_t.index = pd.to_numeric(sample_df_t.index, errors='coerce')
    sample_df_t = sample_df_t[sample_df_t.index.notnull()]
    sample_df_t = sample_df_t.sort_index()

    fig = go.Figure()

    for col in sample_df_t.columns:
        fig.add_trace(go.Scatter(
            x=sample_df_t.index, 
            y=sample_df_t[col], 
            mode='lines', 
            name=str(col),
            line=dict(width=2),
            connectgaps=True
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color='black', size=16)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title=dict(text='<b>Wavelength (nm)</b>', font=dict(color='black', size=14)),
            tickfont=dict(color='black', size=12),
            tickmode='linear',
            tick0=380,
            dtick=200,
            range=[350, 2400],
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            title=dict(text='<b>Reflectance</b>', font=dict(color='black', size=14)),
            tickfont=dict(color='black', size=12),
            tickmode='linear',
            tick0=0,
            dtick=0.1,
            range=[0, 0.8], # Adjust this if your reflectance values exceed 0.8
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        legend=dict(
            title=None,
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black', borderwidth=1,
            font=dict(color='black', size=11)
        ),
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Session State Initialization ---
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'df_resampled' not in st.session_state:
    st.session_state.df_resampled = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# --- Page Header ---
st.title("Step 1: Data Preprocessing Pipeline")
st.write("Upload your raw hyperspectral data to begin the automated preprocessing workflow.")

# ==========================================
# STAGE 1: RAW DATA INPUT & PREVIEW
# ==========================================
uploaded_file = st.file_uploader("Upload your RAW dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Reset pipeline if a NEW file is uploaded
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.raw_data = None
            st.session_state.df_clean = None
            st.session_state.df_resampled = None
            st.session_state.last_uploaded_file = uploaded_file.name
            
            if uploaded_file.name.endswith('.csv'):
                st.session_state.raw_data = pd.read_csv(uploaded_file)
            else:
                st.session_state.raw_data = pd.read_excel(uploaded_file)

        st.success(f"Successfully loaded: {uploaded_file.name}")
        
        st.write("### 1. Raw Data Preview")
        st.dataframe(st.session_state.raw_data.head())
        plot_spectra(st.session_state.raw_data, "Raw Data Spectral Signatures (1 Sample per Cultivar)")
        
        st.markdown("---")

        # ==========================================
        # STAGE 2: NOISE REMOVAL
        # ==========================================
        st.write("### 2. Noise Removal")
        st.write("Remove noisy bands (e.g., < 400 nm and water absorption regions).")
        
        if st.button("Apply Noise Removal"):
            with st.spinner('Applying noise removal...'):
                df = st.session_state.raw_data
                if df.shape[1] < 2:
                    st.error("Data must have at least two columns (one identifier, one band).")
                else:
                    samples = df.iloc[:, 0]
                    X = df.iloc[:, 1:].copy()
                    
                    X.columns = X.columns.astype(float)
                    keep1 = X.loc[:, (X.columns >= 400) & (X.columns <= 1349)]
                    keep2 = X.loc[:, (X.columns >= 1451) & (X.columns <= 1799)]
                    keep3 = X.loc[:, (X.columns >= 1951) & (X.columns <= 2300)]

                    clean_features = pd.concat([keep1, keep2, keep3], axis=1)
                    clean_features.columns = [str(int(x)) for x in clean_features.columns]
                    
                    st.session_state.df_clean = pd.concat([samples, clean_features], axis=1)

        # If Noise Removal is complete, display results and unlock next stage
        if st.session_state.df_clean is not None:
            st.success("Noise removal complete!")
            
            original_band_count = st.session_state.raw_data.shape[1] - 1
            clean_band_count = st.session_state.df_clean.shape[1] - 1
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Bands", original_band_count)
            col2.metric("Bands After Cleaning", clean_band_count)
            col3.metric("Bands Removed", original_band_count - clean_band_count)

            st.write("### Cleaned Data Preview")
            st.dataframe(st.session_state.df_clean.head())
            plot_spectra(st.session_state.df_clean, "Cleaned Data Spectral Signatures (Noisy Bands Removed)")
            
            st.markdown("---")

            # ==========================================
            # STAGE 3: RESAMPLING TO 10 NM
            # ==========================================
            st.write("### 3. Resampling to 10 nm")
            st.write("Resample the cleaned wavelengths to a consistent 10 nm interval.")
            
            if st.button("Apply Resampling to 10nm"):
                with st.spinner('Resampling data...'):
                    samples = st.session_state.df_clean.iloc[:, 0]
                    X = st.session_state.df_clean.iloc[:, 1:].copy()
                    X.columns = X.columns.astype(float)

                    selected = []
                    for wl in range(400, 2451, 10):
                        if not X.empty and X.columns.min() <= wl <= X.columns.max() + 10:
                            nearest = min(X.columns, key=lambda x: abs(x - wl))
                            selected.append(nearest)

                    selected = sorted(list(set(selected)))
                    X10 = X[selected]
                    X10.columns = [str(int(i)) for i in selected]

                    st.session_state.df_resampled = pd.concat([samples, X10], axis=1)

            # If Resampling is complete, display results and unlock splitting stage
            if st.session_state.df_resampled is not None:
                st.success(f"Resampling complete! Reduced to {st.session_state.df_resampled.shape[1] - 1} bands.")
                
                st.write("### Resampled Data Preview")
                st.dataframe(st.session_state.df_resampled.head())
                plot_spectra(st.session_state.df_resampled, "Resampled Data (10 nm) Spectral Signatures")
                
                st.markdown("---")

                # ==========================================
                # STAGE 4: DATASET SPLITTING
                # ==========================================
                st.write("### 4. Dataset Splitting")
                
                split_ratio = st.selectbox(
                    "Select Train:Validation:Test split ratio:",
                    options=["60:20:20", "70:15:15", "80:10:10", "50:25:25", "70:20:10"],
                    index=0 
                )

                if st.button("Split Dataset"):
                    with st.spinner('Splitting data...'):
                        train_pct, val_pct, test_pct = [int(i) for i in split_ratio.split(":")]
                        
                        first_split_test_size = (val_pct + test_pct) / 100.0
                        second_split_test_size = test_pct / (val_pct + test_pct)

                        try:
                            train_df, temp_df = train_test_split(
                                st.session_state.df_resampled, 
                                test_size=first_split_test_size, 
                                random_state=42, 
                                stratify=st.session_state.df_resampled.iloc[:, 0]
                            )

                            val_df, test_df = train_test_split(
                                temp_df, 
                                test_size=second_split_test_size, 
                                random_state=42, 
                                stratify=temp_df.iloc[:, 0]
                            )

                            st.success(f"Data successfully split using {split_ratio} ratio!")

                            col1, col2, col3 = st.columns(3)
                            col1.metric(f"Training Set ({train_pct}%)", f"{train_df.shape[0]} rows")
                            col2.metric(f"Validation Set ({val_pct}%)", f"{val_df.shape[0]} rows")
                            col3.metric(f"Testing Set ({test_pct}%)", f"{test_df.shape[0]} rows")

                            st.write("Download your prepared datasets as CSV files below:")
                            
                            col_dl1, col_dl2, col_dl3 = st.columns(3)

                            col_dl1.download_button(label="📥 Download Train Data", data=train_df.to_csv(index=False).encode('utf-8'), file_name='train_data_10nm.csv', mime='text/csv')
                            col_dl2.download_button(label="📥 Download Validation Data", data=val_df.to_csv(index=False).encode('utf-8'), file_name='val_data_10nm.csv', mime='text/csv')
                            col_dl3.download_button(label="📥 Download Test Data", data=test_df.to_csv(index=False).encode('utf-8'), file_name='test_data_10nm.csv', mime='text/csv')
                            
                        except ValueError as e:
                            st.error(f"Splitting failed. This usually happens if your classes are too small to properly stratify. Try a different ratio. \n\nDetailed Error: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
