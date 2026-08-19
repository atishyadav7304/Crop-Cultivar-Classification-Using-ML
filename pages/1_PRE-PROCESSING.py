import plotly.graph_objects as go

# --- Helper Function for Plotting Spectral Data ---
def plot_spectra(data, title):
    """
    Groups data by the first column (cultivar), takes 1 sample per cultivar, 
    and plots Reflectance vs Wavelength mimicking a specific scientific layout.
    """
    # Group by the first column (cultivar) and get the first row for each
    sample_df = data.groupby(data.columns[0]).first()
    
    # Force all reflectance values to be numeric
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
    
    # Transpose so that wavelengths become the index
    sample_df_t = sample_df.T
    
    # Convert index (wavelengths) to numeric values
    sample_df_t.index = pd.to_numeric(sample_df_t.index, errors='coerce')
    
    # Drop invalid rows and sort
    sample_df_t = sample_df_t[sample_df_t.index.notnull()]
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
        
        # X-Axis Styling
        xaxis=dict(
            title='<b>Wavelength (nm)</b>',
            titlefont=dict(color='black', size=14),
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
        
        # Y-Axis Styling
        yaxis=dict(
            title='<b>Reflectance</b>',
            titlefont=dict(color='black', size=14),
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
