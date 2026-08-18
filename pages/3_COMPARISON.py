import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Step 3: Model Comparison & Leaderboard")
st.write("Compare the predictive performance of all trained models to select the best algorithm for hyperspectral cultivar discrimination.")

if 'model_results' in st.session_state and len(st.session_state.model_results) > 0:
    # Convert session state results to a DataFrame
    results_df = pd.DataFrame(st.session_state.model_results)
    
    # Sort by Accuracy descending
    results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    st.write("### Model Leaderboard")
    # Format the dataframe to display percentages nicely
    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "Kappa"]:
        display_df[col] = (display_df[col] * 100).apply(lambda x: f"{x:.2f}%")
        
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    st.write("### Performance Visualization")
    
    # Create a bar chart comparing Accuracy and F1 Score
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(results_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], results_df["Accuracy"], width, label='Accuracy', color='#4CAF50')
    ax.bar([i + width/2 for i in x], results_df["F1 Score"], width, label='F1 Score', color='#2196F3')
    
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=45, ha='right')
    ax.legend()
    
    # Display plot
    st.pyplot(fig)
    
    # Identify the best model
    best_model = results_df.iloc[0]
    st.success(f"🏆 **Best Performing Model:** {best_model['Model']} with an Accuracy of {best_model['Accuracy']*100:.2f}%")

else:
    st.info("No models have been trained yet. Please go to the 'Modeling' page, upload your data, and train at least one model to see the comparison.")