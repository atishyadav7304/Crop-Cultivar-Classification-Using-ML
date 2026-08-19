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
            range=[350, 2500],
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
            range=[-1, 3.5], # Adjust this if your reflectance values exceed 0.8
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

# --- Page Header ---
st.title("Step 1: Data Preprocessing")
st.write("Select your processing stage below, then upload the corresponding dataset to begin.")

# --- Dropdown Menu for Workflow Steps ---
processing_step = st.selectbox(
    "Select Processing Stage:",
    options=[
        "1. Raw Data Input", 
        "2. Noise Removal Step", 
        "3. Resampling & Data Splitting Step"
    ],
    index=0
)

st.markdown("---")

# ==========================================
# STAGE 1: RAW DATA INPUT
# ==========================================
if processing_step == "1. Raw Data Input":
    st.write("### 📤 Upload Raw Data")
    raw_file = st.file_uploader("Upload your RAW dataset (.csv or .xlsx)", type=["csv", "xlsx"], key="raw")
    
    if raw_file is not None:
        try:
            if raw_file.name.endswith('.csv'):
                df = pd.read_csv(raw_file)
            else:
                df = pd.read_excel(raw_file)
                
            st.success(f"Successfully loaded: {raw_file.name}")
            
            st.write("### Raw Data Preview")
            st.dataframe(df.head())
            plot_spectra(df, "Raw Data Spectral Signatures (1 Sample per Cultivar)")
            
            st.markdown("---")
            st.write("### Apply Next Step: Noise Removal")
            st.write("Remove noisy bands (e.g., < 400 nm and water absorption regions).")
            
            if st.button("Apply Noise Removal"):
                with st.spinner('Applying noise removal...'):
                    if df.shape[1] < 2:
                        st.error("Data must have at least two columns (one identifier, one band).")
                    else:
                        samples = df.iloc[:, 0]
                        X = df.iloc[:, 1:].copy()
                        original_band_count = X.shape[1]

                        X.columns = X.columns.astype(float)
                        keep1 = X.loc[:, (X.columns >= 400) & (X.columns <= 1349)]
                        keep2 = X.loc[:, (X.columns >= 1451) & (X.columns <= 1799)]
                        keep3 = X.loc[:, (X.columns >= 1951) & (X.columns <= 2300)]

                        clean_features = pd.concat([keep1, keep2, keep3], axis=1)
                        clean_features.columns = [str(int(x)) for x in clean_features.columns]
                        
                        df_clean = pd.concat([samples, clean_features], axis=1)
                        
                        st.success("Noise removal complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Original Bands", original_band_count)
                        col2.metric("Bands After Cleaning", clean_features.shape[1])
                        col3.metric("Bands Removed", original_band_count - clean_features.shape[1])

                        st.write("### Cleaned Data Preview")
                        st.dataframe(df_clean.head())
                        plot_spectra(df_clean, "Cleaned Data Spectral Signatures (Noisy Bands Removed)")
                        
                        # Provide a download button for the intermediate step
                        st.download_button(
                            label="📥 Download Noise-Removed Data",
                            data=df_clean.to_csv(index=False).encode('utf-8'),
                            file_name='noise_removed_data.csv',
                            mime='text/csv'
                        )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ==========================================
# STAGE 2: NOISE REMOVAL STEP
# ==========================================
elif processing_step == "2. Noise Removal Step":
    st.write("### 📤 Upload Noise-Removed Data")
    clean_file = st.file_uploader("Upload your NOISE-REMOVED dataset (.csv or .xlsx)", type=["csv", "xlsx"], key="clean")
    
    if clean_file is not None:
        try:
            if clean_file.name.endswith('.csv'):
                df_clean = pd.read_csv(clean_file)
            else:
                df_clean = pd.read_excel(clean_file)
                
            st.success(f"Successfully loaded: {clean_file.name}")
            
            st.write("### Cleaned Data Preview")
            st.dataframe(df_clean.head())
            plot_spectra(df_clean, "Cleaned Data Spectral Signatures (Noisy Bands Removed)")
            
            st.markdown("---")
            st.write("### Apply Next Step: Resampling to 10 nm")
            st.write("Resample the cleaned wavelengths to a consistent 10 nm interval.")
            
            if st.button("Apply Resampling to 10nm"):
                with st.spinner('Resampling data...'):
                    samples = df_clean.iloc[:, 0]
                    X = df_clean.iloc[:, 1:].copy()
                    X.columns = X.columns.astype(float)

                    selected = []
                    for wl in range(400, 2451, 10):
                        if not X.empty and X.columns.min() <= wl <= X.columns.max() + 10:
                            nearest = min(X.columns, key=lambda x: abs(x - wl))
                            selected.append(nearest)

                    selected = sorted(list(set(selected)))
                    X10 = X[selected]
                    X10.columns = [str(int(i)) for i in selected]

                    df_resampled = pd.concat([samples, X10], axis=1)

                    st.success(f"Resampling complete! Reduced to {X10.shape[1]} bands.")
                    
                    st.write("### Resampled Data Preview")
                    st.dataframe(df_resampled.head())
                    plot_spectra(df_resampled, "Resampled Data (10 nm) Spectral Signatures")
                    
                    # Provide a download button for the resampled step
                    st.download_button(
                        label="📥 Download Resampled Data (10nm)",
                        data=df_resampled.to_csv(index=False).encode('utf-8'),
                        file_name='resampled_10nm_data.csv',
                        mime='text/csv'
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ==========================================
# STAGE 3: RESAMPLING & SPLITTING
# ==========================================
elif processing_step == "3. Resampling & Data Splitting Step":
    st.write("### 📤 Upload Resampled Data")
    resampled_file = st.file_uploader("Upload your RESAMPLED dataset (.csv or .xlsx)", type=["csv", "xlsx"], key="resampled")
    
    if resampled_file is not None:
        try:
            if resampled_file.name.endswith('.csv'):
                df_resampled = pd.read_csv(resampled_file)
            else:
                df_resampled = pd.read_excel(resampled_file)
                
            st.success(f"Successfully loaded: {resampled_file.name}")
            
            st.write("### Resampled Data Preview")
            st.dataframe(df_resampled.head())
            plot_spectra(df_resampled, "Resampled Data (10 nm) Spectral Signatures")
            
            st.markdown("---")
            st.write("### Apply Next Step: Dataset Splitting")
            
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
                            df_resampled, 
                            test_size=first_split_test_size, 
                            random_state=42, 
                            stratify=df_resampled.iloc[:, 0]
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
