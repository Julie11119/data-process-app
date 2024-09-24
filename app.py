# app.py

import streamlit as st
import pandas as pd
import numpy as np
from openai_api import get_cleaning_suggestions, get_visualization_suggestions
from data_processing import (load_data, generate_data_summary, clean_data, perform_eda, 
                             suggest_visualizations, build_initial_model, generate_narrative_insights)
from utils.visualization import (create_histogram, create_scatter_plot, create_box_plot,
                                 create_heatmap, create_pie_chart, create_choropleth)
from utils.helpers import identify_key_columns
import time

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
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    # Determine file type
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        # Load data
        with st.spinner('Loading data...'):
            df = load_data(uploaded_file, file_type)
            time.sleep(1)  # Simulate processing time
        st.success('Data loaded successfully!')
        
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())
        
        # Generate data summary
        st.subheader("📝 Dataset Summary")
        data_summary = generate_data_summary(df)
        st.text(data_summary)
        
        # Get cleaning suggestions from OpenAI
        st.subheader("💡 Data Cleaning Suggestions")
        cleaning_suggestions = get_cleaning_suggestions(data_summary)
        st.write(cleaning_suggestions)
        
        # Clean data based on suggestions
        with st.spinner('Cleaning data based on suggestions...'):
            df_cleaned = clean_data(df, cleaning_suggestions)
            time.sleep(1)  # Simulate processing time
        st.success('Data cleaning completed!')
        
        st.subheader("🧼 Cleaned Data Preview")
        st.dataframe(df_cleaned.head())
        
        # Generate summary of cleaned data
        st.subheader("📝 Cleaned Data Summary")
        cleaned_summary = generate_data_summary(df_cleaned)
        st.text(cleaned_summary)
        
        # Get visualization suggestions from OpenAI
        st.subheader("🎨 Visualization Suggestions")
        visualization_suggestions = get_visualization_suggestions(cleaned_summary)
        st.write(visualization_suggestions)
        
        # Suggest visualization types
        suggested_viz = suggest_visualizations(df_cleaned, visualization_suggestions)
        st.subheader("✅ Suggested Visualizations")
        st.write(", ".join(suggested_viz))
        
        # Perform EDA
        st.subheader("📊 Exploratory Data Analysis")
        eda_results = perform_eda(df_cleaned)
        
        # Display EDA visualizations
        for key, fig in eda_results.items():
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional Visualizations Based on Suggestions
        st.subheader("🔍 Additional Visualizations")
        for viz in suggested_viz:
            if viz.lower() == "histogram":
                numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select column for Histogram", numeric_cols, key="hist_select")
                    fig = create_histogram(df_cleaned, selected_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for Histogram.")
            
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
                    st.warning("Not enough numeric columns available for Scatter Plot.")
            
            elif viz.lower() == "box plot":
                numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
                if len(numeric_cols) >=1 and len(categorical_cols) >=1:
                    numeric_col = st.selectbox("Select Numeric Column for Box Plot", numeric_cols, key="box_numeric")
                    category_col = st.selectbox("Select Categorical Column for Box Plot", categorical_cols, key="box_category")
                    fig = create_box_plot(df_cleaned, numeric_col, category_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough columns available for Box Plot.")
            
            elif viz.lower() == "pie chart":
                categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    selected_col = st.selectbox("Select Categorical Column for Pie Chart", categorical_cols, key="pie_select")
                    fig = create_pie_chart(df_cleaned, selected_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No categorical columns available for Pie Chart.")
            
            elif viz.lower() == "choropleth map":
                # Assuming there's a 'country' column and a 'total_amount' column
                if 'country' in df_cleaned.columns and 'total_amount' in df_cleaned.columns:
                    fig = create_choropleth(df_cleaned, 'country', 'total_amount')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Choropleth Map requires 'country' and 'total_amount' columns.")
            
            # Add more visualization types as needed
    
        # Option to download cleaned data
        st.subheader("📥 Download Cleaned Data")
        csv_cleaned = df_cleaned.to_csv(index=False)
        st.download_button(label='Download Cleaned CSV', data=csv_cleaned, file_name='cleaned_data.csv', mime='text/csv')
        
        # Sidebar: Natural Language Queries
        st.sidebar.header('2. Ask a Question')
        user_query = st.sidebar.text_input("Enter your question about the data:")
        
        if user_query:
            with st.spinner('Processing your query...'):
                # Generate a prompt for OpenAI
                prompt = f"""
                You are a data analyst assistant. Given the following dataset summary, answer the user's question with appropriate data analysis steps or visualizations.

                Dataset Summary:
                {cleaned_summary}

                User Question:
                {user_query}

                Response:
                """
                
                # Fetch response from OpenAI
                response = get_cleaning_suggestions(prompt)  # Reuse cleaning_suggestions function for simplicity
                answer = response  # Extract answer from response
                
                # Note: For more accurate natural language processing, consider implementing a separate function.
            
            st.subheader("💬 Response to Your Query")
            st.write(answer)
        
        # Sidebar: Machine Learning
        st.sidebar.header('3. Machine Learning')
        
        # Identify target columns (numeric columns)
        target_options = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if len(target_options) > 0:
            target_column = st.sidebar.selectbox("Select Target Column for Prediction", target_options)
            
            if st.sidebar.button('Build Initial Model'):
                with st.spinner('Building model...'):
                    model_results = build_initial_model(df_cleaned, target_column)
                    time.sleep(1)  # Simulate processing time
                st.success('Model built successfully!')
                st.write(f"**Mean Squared Error:** {model_results['mse']}")
        else:
            st.sidebar.warning("No numeric columns available for machine learning.")
        
        # Sidebar: Automated Report Generation
        st.sidebar.header('4. Generate Report')
        
        if st.sidebar.button('Generate Narrative Insights'):
            with st.spinner('Generating report...'):
                narrative_insights = generate_narrative_insights(df_cleaned, eda_results)
                time.sleep(1)  # Simulate processing time
            st.subheader("📄 Narrative Insights")
            st.write(narrative_insights)
        
    else:
        st.info("📥 Please upload a dataset to begin.")
