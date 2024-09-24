# app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from openai_api import (
    get_cleaning_suggestions,
    get_visualization_suggestions,
    get_narrative_response
)
from data_processing import (
    load_data,
    generate_data_summary,
    perform_eda,
    suggest_visualizations,
    build_initial_model,
    generate_narrative_insights
)
from visualization import (
    create_histogram,
    create_scatter_plot,
    create_box_plot,
    create_heatmap,
    create_pie_chart,
    create_choropleth
)
from helpers import identify_key_columns
import time
import os

# Streamlit app configuration
st.set_page_config(
    page_title="Advanced Data Preparation & EDA App",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title('🧹 Advanced Data Preparation & EDA App')

# Sidebar for file upload and other settings
st.sidebar.header('1. Upload Your Dataset')

# File uploader supporting CSV, Excel, JSON
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "xlsm", "json"])

if uploaded_file is not None:
    # Determine file type using robust method
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()  # e.g., '.xlsx'
    file_type = file_extension[1:]  # e.g., 'xlsx'
    
    st.write(f"**Detected file type:** `{file_type}`")  # Debugging statement
    
    try:
        # Load data
        with st.spinner('Loading data...'):
            df = load_data(uploaded_file, file_type)
            time.sleep(1)  # Simulate processing time
        st.success('✅ Data loaded successfully!')
        
        # Display DataFrame Information
        st.subheader("🗂️ Raw Data")
        st.dataframe(df.head())

        # Generate a summary of the dataset
        st.subheader("📝 Dataset Summary")
        df_summary = df.describe(include='all').to_string()
        data_summary = generate_data_summary(df)
        st.text(df_summary)

        # Get cleaning suggestions from OpenAI
        with st.spinner("💡 Getting cleaning suggestions from OpenAI..."):
            suggestions = get_cleaning_suggestions(df_summary)
        st.subheader("💡 Cleaning Suggestions")
        if suggestions:
            try:
                suggestions_json = json.loads(suggestions)
                st.json(suggestions_json)
            except json.JSONDecodeError:
                st.error("⚠️ Failed to parse cleaning suggestions. Please ensure the response is in valid JSON format.")
                st.text(suggestions)
                suggestions_json = {}
        else:
            st.warning("⚠️ No cleaning suggestions received.")
            suggestions_json = {}

    except ValueError as ve:
        st.error(f"⚠️ Value Error: {ve}. Please ensure you're uploading a supported file type (CSV, Excel, JSON).")
        suggestions_json = {}
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred while loading the data: {e}")
        suggestions_json = {}
    else:
        if suggestions_json:
            # Apply the cleaning suggestions
            try:
                with st.spinner("🧼 Cleaning data based on suggestions..."):
                    df_cleaned = cleaning_suggestions(df.copy(), suggestions_json)
                st.success("✅ Data cleaning completed!")
            except Exception as e:
                st.error(f"⚠️ An error occurred while cleaning the data: {e}")
                df_cleaned = df.copy()

            st.subheader("🗂️ Cleaned Data")
            st.dataframe(df_cleaned.head())
            df_cleaned_summary = df_cleaned.describe(include='all').to_string()
            st.text(df_cleaned_summary)
            
            # Get visualization suggestions from OpenAI
            try:
                with st.spinner("🎨 Getting visualization suggestions from OpenAI..."):
                    viz_suggestions = get_visualization_suggestions(df_cleaned_summary)
                st.subheader("🎨 Visualization Suggestions")
                if viz_suggestions:
                    st.write(", ".join(viz_suggestions))
                else:
                    st.warning("⚠️ No visualization suggestions received.")
            except Exception as e:
                st.error(f"⚠️ An error occurred while getting visualization suggestions: {e}")
                viz_suggestions = []

            # Suggest visualization types
            if viz_suggestions:
                try:
                    suggested_viz = suggest_visualizations(df_cleaned, viz_suggestions)
                    st.subheader("✅ Suggested Visualizations")
                    st.write(", ".join(suggested_viz))
                except Exception as e:
                    st.error(f"⚠️ An error occurred while suggesting visualizations: {e}")
                    suggested_viz = []
            else:
                suggested_viz = []
            
            # Perform EDA
            try:
                st.subheader("📊 Exploratory Data Analysis")
                eda_results = perform_eda(df_cleaned)
                
                if eda_results:
                    # Display Correlation Matrix
                    st.write("**Correlation Matrix:**")
                    st.dataframe(eda_results.get('correlation_matrix', pd.DataFrame()))
                    
                    # Display Correlation Heatmap
                    if 'correlation_matrix' in eda_results:
                        st.write("**Correlation Heatmap:**")
                        fig = create_heatmap(eda_results['correlation_matrix'])
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ An error occurred during EDA: {e}")
            
            # Additional Visualizations Based on Suggestions
            if suggested_viz:
                try:
                    st.subheader("🔍 Additional Visualizations")
                    for viz in suggested_viz:
                        if viz.lower() == "histogram":
                            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                selected_col = st.selectbox("Select column for Histogram", numeric_cols, key="hist_select")
                                fig = create_histogram(df_cleaned, selected_col)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ No numeric columns available for Histogram.")
                        
                        elif viz.lower() == "scatter plot":
                            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                            categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
                            if len(numeric_cols) >= 2:
                                x_col = st.selectbox("Select X-axis for Scatter Plot", numeric_cols, key="scatter_x")
                                y_col = st.selectbox("Select Y-axis for Scatter Plot", numeric_cols, key="scatter_y")
                                color_col = st.selectbox("Select Color by (Optional)", [None] + list(categorical_cols), key="scatter_color")
                                fig = create_scatter_plot(df_cleaned, x_col, y_col, color_col)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ Not enough numeric columns available for Scatter Plot.")
                        
                        elif viz.lower() == "box plot":
                            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                            categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
                            if len(numeric_cols) >=1 and len(categorical_cols) >=1:
                                numeric_col = st.selectbox("Select Numeric Column for Box Plot", numeric_cols, key="box_numeric")
                                category_col = st.selectbox("Select Categorical Column for Box Plot", categorical_cols, key="box_category")
                                fig = create_box_plot(df_cleaned, numeric_col, category_col)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ Not enough columns available for Box Plot.")
                        
                        elif viz.lower() == "pie chart":
                            categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
                            if len(categorical_cols) > 0:
                                selected_col = st.selectbox("Select Categorical Column for Pie Chart", categorical_cols, key="pie_select")
                                fig = create_pie_chart(df_cleaned, selected_col)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ No categorical columns available for Pie Chart.")
                        
                        elif viz.lower() == "choropleth map":
                            # Assuming there's a 'country' column and a 'total_amount' column
                            if 'country' in df_cleaned.columns and 'total_amount' in df_cleaned.columns:
                                fig = create_choropleth(df_cleaned, 'country', 'total_amount')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ Choropleth Map requires 'country' and 'total_amount' columns.")
                        
                        # Add more visualization types as needed
                except Exception as e:
                    st.error(f"⚠️ An error occurred while creating visualizations: {e}")
            
            # Option to download cleaned data
            try:
                st.subheader("📥 Download Cleaned Data")
                csv_cleaned = df_cleaned.to_csv(index=False)
                st.download_button(
                    label='📥 Download Cleaned CSV',
                    data=csv_cleaned,
                    file_name='cleaned_data.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"⚠️ An error occurred while preparing the download: {e}")
            
            # Sidebar: Natural Language Queries
            st.sidebar.header('2. Ask a Question')
            user_query = st.sidebar.text_input("Enter your question about the data:")
            
            if user_query:
                try:
                    with st.spinner('🗣️ Processing your query...'):
                        # Generate a prompt for OpenAI
                        prompt = f"""
                        You are a data analyst assistant. Given the following dataset summary, answer the user's question with appropriate data analysis steps or visualizations.
        
                        Dataset Summary:
                        {df_cleaned.describe(include='all').to_string()}
        
                        User Question:
                        {user_query}
        
                        Response:
                        """
                        
                        # Fetch response from OpenAI using the new function
                        answer = get_narrative_response(prompt)
                    
                    st.subheader("💬 Response to Your Query")
                    st.write(answer)
                except Exception as e:
                    st.error(f"⚠️ An error occurred while processing your query: {e}")
            
            # Sidebar: Machine Learning
            st.sidebar.header('3. Machine Learning')
            
            # Identify target columns (numeric columns)
            target_options = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            if len(target_options) > 0:
                target_column = st.sidebar.selectbox("Select Target Column for Prediction", target_options)
                
                if st.sidebar.button('🚀 Build Initial Model'):
                    try:
                        with st.spinner('🔧 Building model...'):
                            model_results = build_initial_model(df_cleaned, target_column)
                            time.sleep(1)  # Simulate processing time
                        if model_results:
                            st.success('✅ Model built successfully!')
                            st.write(f"**Mean Squared Error:** {model_results['mse']}")
                    except Exception as e:
                        st.error(f"⚠️ An error occurred while building the model: {e}")
            else:
                st.sidebar.warning("⚠️ No numeric columns available for machine learning.")
            
            # Sidebar: Automated Report Generation
            st.sidebar.header('4. Generate Report')
            
            if st.sidebar.button('📄 Generate Narrative Insights'):
                try:
                    with st.spinner('📑 Generating report...'):
                        narrative_insights = generate_narrative_insights(df_cleaned, eda_results)
                        time.sleep(1)  # Simulate processing time
                    st.subheader("📄 Narrative Insights")
                    st.write(narrative_insights)
                except Exception as e:
                    st.error(f"⚠️ An error occurred while generating the report: {e}")
        
        else:
            st.warning("⚠️ Unable to clean data without valid suggestions.")
else:
    st.info("📥 Please upload a dataset to begin.")
