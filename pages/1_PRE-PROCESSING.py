import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Session State Initialization ---
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'df_resampled' not in st.session_state:
    st.session_state.df_resampled = None

st.title("Step 1: Data Preprocessing")
st.write("Upload your raw hyperspectral data to begin noise removal, resampling, and data splitting.")

# --- 1. File Uploader ---
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            
        st.success(f"Successfully loaded: {uploaded_file.name}")
        st.write("### Raw Data Preview")
        st.dataframe(df.head())

        st.markdown("---")

        # --- 2. Noise Removal Section ---
        st.write("### Noise Removal")
        st.write("Click below to remove noisy bands (e.g., < 400 nm and water absorption regions).")
        
        if st.button("Apply Noise Removal"):
            with st.spinner('Applying noise removal...'):
                try:
                    # Your specific noise removal logic
                    if df.shape[1] < 2:
                        st.error("Data must have at least two columns (one identifier, one band).")
                        st.stop()

                    samples = df.iloc[:, 0]
                    X = df.iloc[:, 1:].copy()
                    original_band_count = X.shape[1]

                    # Convert column headers to float for filtering
                    X.columns = X.columns.astype(float)

                    # Keep only the useful spectral regions
                    keep1 = X.loc[:, (X.columns >= 400) & (X.columns <= 1349)]
                    keep2 = X.loc[:, (X.columns >= 1451) & (X.columns <= 1799)]
                    keep3 = X.loc[:, (X.columns >= 1951) & (X.columns <= 2300)]

                    # Concatenate the kept regions
                    clean_features = pd.concat([keep1, keep2, keep3], axis=1)
                    
                    # Convert column names back to integer strings for clarity
                    clean_features.columns = [str(int(x)) for x in clean_features.columns]

                    # Re-combine the sample identifiers with the cleaned features
                    df_clean = pd.concat([samples, clean_features], axis=1)
                    
                    # Store in session state
                    st.session_state.df_clean = df_clean

                    st.success("Noise removal complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Bands", original_band_count)
                    col2.metric("Bands After Cleaning", clean_features.shape[1])
                    col3.metric("Bands Removed", original_band_count - clean_features.shape[1])
                    
                    st.write("### Cleaned Data Preview")
                    st.dataframe(df_clean.head())
                    
                except Exception as e:
                    st.error(f"An error occurred during noise removal: {e}")

        # --- 3. Resampling Section ---
        if st.session_state.df_clean is not None:
            st.markdown("---")
            st.write("### Resampling to 10 nm")
            st.write("Click below to resample the cleaned wavelengths to a 10 nm interval.")
            
            if st.button("Apply Resampling"):
                with st.spinner('Resampling data...'):
                    try:
                        # Your specific resampling logic
                        df_clean = st.session_state.df_clean
                        samples = df_clean.iloc[:, 0]
                        X = df_clean.iloc[:, 1:].copy()
                        X.columns = X.columns.astype(float)
                        
                        selected = []
                        
                        for wl in range(400, 2451, 10):
                            # Ensure we don't try to find a wavelength way outside our cleaned columns
                            if not X.empty and X.columns.min() <= wl <= X.columns.max() + 10:
                                nearest = min(X.columns, key=lambda x: abs(x - wl))
                                selected.append(nearest)
                        
                        selected = sorted(list(set(selected)))
                        X10 = X[selected]
                        X10.columns = [str(int(i)) for i in selected]
                        
                        df_resampled = pd.concat([samples, X10], axis=1)
                        st.session_state.df_resampled = df_resampled
                        
                        st.success(f"Resampling complete! Reduced to {X10.shape[1]} bands.")
                        st.write("### Resampled Data Preview")
                        st.dataframe(df_resampled.head())
                        
                    except Exception as e:
                        st.error(f"An error occurred during resampling: {e}")

        # --- 4. Data Splitting & Export Section ---
        if st.session_state.df_resampled is not None:
            st.markdown("---")
            st.write("### Dataset Splitting (60:20:20)")
            st.write("Split the resampled dataset into Training, Validation, and Testing sets.")
            
            if st.button("Split Dataset"):
                with st.spinner('Splitting data...'):
                    df_res = st.session_state.df_resampled
                    
                    # Split into 60% Train, 40% Temp (Validation + Test)
                    train_df, temp_df = train_test_split(
                        df_res, test_size=0.40, random_state=42, stratify=df_res.iloc[:, 0]
                    )
                    
                    # Split Temp into 50% Validation, 50% Test (which equals 20% of total each)
                    val_df, test_df = train_test_split(
                        temp_df, test_size=0.50, random_state=42, stratify=temp_df.iloc[:, 0]
                    )
                    
                    st.success("Data successfully split!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Training Set (60%)", f"{train_df.shape[0]} rows")
                    col2.metric("Validation Set (20%)", f"{val_df.shape[0]} rows")
                    col3.metric("Testing Set (20%)", f"{test_df.shape[0]} rows")
                    
                    st.write("Download your prepared datasets as CSV files below:")
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    col_dl1.download_button(
                        label="Download Train Data",
                        data=train_df.to_csv(index=False).encode('utf-8'),
                        file_name='train_data_10nm.csv',
                        mime='text/csv'
                    )
                    
                    col_dl2.download_button(
                        label="Download Validation Data",
                        data=val_df.to_csv(index=False).encode('utf-8'),
                        file_name='val_data_10nm.csv',
                        mime='text/csv'
                    )
                    
                    col_dl3.download_button(
                        label="Download Test Data",
                        data=test_df.to_csv(index=False).encode('utf-8'),
                        file_name='test_data_10nm.csv',
                        mime='text/csv'
                    )
                    
    except Exception as e:
        st.error(f"An error occurred: {e}")
