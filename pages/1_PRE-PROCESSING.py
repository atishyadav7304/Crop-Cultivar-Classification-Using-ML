import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go  # <-- NEW: Changed to graph_objects for advanced styling

# --- Helper Function for Plotting Spectral Data ---
def plot_spectra(data, title):
    """
    Groups data by the first column (cultivar), takes 1 sample per cultivar, 
    and plots Reflectance vs Wavelength mimicking the academic scientific layout.
    """
    # Group by the first column (target variable/cultivar) and get the first row for each
    sample_df = data.groupby(data.columns[0]).first()
    
    # Force all reflectance values to be numeric to avoid categorical axis issues
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
    
    # Transpose so that wavelengths become the index, and cultivars become columns
    sample_df_t = sample_df.T
    
    # Convert index (wavelengths) to numeric values for proper x-axis scaling
    sample_df_t.index = pd.to_numeric(sample_df_t.index, errors='coerce')
    
    # Drop any rows where the index isn't a valid number
    sample_df_t = sample_df_t[sample_df_t.index.notnull()]
    
    # Sort by wavelength to ensure lines are drawn sequentially
    sample_df_t = sample_df_t.sort_index()

    # Create the figure using Graph Objects for finer control
    fig = go.Figure()

    # Add a line for each crop cultivar
    for col in sample_df_t.columns:
        fig.add_trace(go.Scatter(
            x=sample_df_t.index, 
            y=sample_df_t[col], 
            mode='lines', 
            name=str(col),
            line=dict(width=2),
            connectgaps=True # Connects the lines across removed noisy bands
        ))

    # Apply the exact styling from the reference image
    fig.update_layout(
        title=dict(text=title, font=dict(color='black', size=16)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        
        # X-Axis Styling (FIXED TITLE FONT)
        xaxis=dict(
            title=dict(
                text='<b>Wavelength (nm)</b>',
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black', size=12),
            tickmode='linear',
            tick0=380,     # Starts ticks exactly at 380
            dtick=200,     # Intervals of 200 (380, 580, 780...)
            range=[350, 2500], # Adds slight padding around the data
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True    # Creates the top border of the box
        ),
        
        # Y-Axis Styling (FIXED TITLE FONT)
        yaxis=dict(
            title=dict(
                text='<b>Reflectance</b>',
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black', size=12),
            tickmode='linear',
            tick0=0,
            dtick=0.1,     # Intervals of 0.1 (0, 0.1, 0.2...)
            range=[-1, 3.5], # Locks the Y-axis from 0 to 0.8
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True    # Creates the right border of the box
        ),
        
        # Legend Styling (Inside top-right with border)
        legend=dict(
            title=None,
            x=0.98,        # Position slightly left of the right edge
            y=0.98,        # Position slightly below the top edge
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)', # Solid white background
            bordercolor='black',
            borderwidth=1,
            font=dict(color='black', size=11)
        ),
        
        # Margins to match the tight look of the image
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="x unified"
    )
    
    # Display in streamlit
    st.plotly_chart(fig, use_container_width=True)


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
        
        # --- NEW: Plot Raw Data ---
        plot_spectra(df, "Raw Data Spectral Signatures (1 Sample per Cultivar)")

        st.markdown("---")

        # --- 2. Noise Removal Section ---
        st.write("### Noise Removal")
        st.write("Click below to remove noisy bands (e.g., < 400 nm and water absorption regions).")

        if st.button("Apply Noise Removal"):
            with st.spinner('Applying noise removal...'):
                try:
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
                    
                    # --- NEW: Plot Cleaned Data ---
                    plot_spectra(df_clean, "Cleaned Data Spectral Signatures (Noisy Bands Removed)")

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
                        
                        # --- NEW: Plot Resampled Data ---
                        plot_spectra(df_resampled, "Resampled Data (10 nm) Spectral Signatures")

                    except Exception as e:
                        st.error(f"An error occurred during resampling: {e}")

        # --- 4. Data Splitting & Export Section ---
        if st.session_state.df_resampled is not None:
            st.markdown("---")
            st.write("### Dataset Splitting")
            
            split_ratio = st.selectbox(
                "Select Train:Validation:Test split ratio:",
                options=["60:20:20", "70:15:15", "80:10:10", "50:25:25", "70:20:10"],
                index=0 # Defaults to 60:20:20
            )

            if st.button("Split Dataset"):
                with st.spinner('Splitting data...'):
                    df_res = st.session_state.df_resampled

                    # Extract percentages from the string (e.g. "60:20:20" -> 60, 20, 20)
                    train_pct, val_pct, test_pct = [int(i) for i in split_ratio.split(":")]
                    
                    # Calculate proper fractions for scikit-learn
                    first_split_test_size = (val_pct + test_pct) / 100.0
                    second_split_test_size = test_pct / (val_pct + test_pct)

                    try:
                        # Split 1: Isolate Training set, and put Validation + Test into temp_df
                        train_df, temp_df = train_test_split(
                            df_res, 
                            test_size=first_split_test_size, 
                            random_state=42, 
                            stratify=df_res.iloc[:, 0]
                        )

                        # Split 2: Divide temp_df into Validation and Test sets
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
                    except ValueError as e:
                        # Catch stratification errors (e.g., if a class has only 1 instance)
                        st.error(f"Splitting failed. This usually happens if your classes are too small to properly stratify based on the selected ratio. Try a different ratio or ensure you have enough data per class. \n\nDetailed Error: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
