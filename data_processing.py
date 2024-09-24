# data_processing.py

import pandas as pd
import numpy as np
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

@st.cache_data(show_spinner=False)
def load_data(uploaded_file, file_type):
    """
    Load data from various file types.
    
    Args:
        uploaded_file (UploadedFile): The uploaded file from Streamlit.
        file_type (str): Type of the file (e.g., 'csv', 'xlsx', 'json').
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        ValueError: If the file type is unsupported or loading fails.
    """
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type in ["xlsx", "xls", "xlsm"]:
            df = pd.read_excel(uploaded_file)
        elif file_type == "json":
            df = pd.read_json(uploaded_file)
        else:
            raise ValueError("Unsupported file type.")
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise ValueError(f"Failed to load data: {e}")

def generate_data_summary(df):
    """
    Generate a textual summary of the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to summarize.
    
    Returns:
        str: Summary of the DataFrame.
    """
    summary = df.describe(include='all').to_string()
    return summary

@st.cache_data(show_spinner=False)
def clean_data(df, cleaning_suggestions):
    """
    Clean the DataFrame based on suggestions.
    
    Args:
        df (pd.DataFrame): The original DataFrame.
        cleaning_suggestions (str): Suggestions for cleaning.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Implement cleaning based on suggestions
    # For simplicity, let's assume we drop rows with missing values
    try:
        df_cleaned = df.dropna()
        logging.info("Data cleaned successfully.")
        return df_cleaned
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        st.error(f"Data cleaning failed: {e}")
        return df

@st.cache_data(show_spinner=False)
def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the cleaned DataFrame.
    
    Args:
        df (pd.DataFrame): The cleaned dataset.
    
    Returns:
        dict: Dictionary containing various EDA results.
    """
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns available for EDA.")
        
        # Compute correlation matrix
        corr = numeric_df.corr()
        
        # Additional EDA tasks can be added here
        
        eda_results = {
            'correlation_matrix': corr,
            # Add other EDA results here
        }
        
        logging.info("EDA performed successfully.")
        return eda_results
    except Exception as e:
        st.error(f"An error occurred during EDA: {e}")
        logging.error(f"EDA error: {e}")
        return {}

def suggest_visualizations(df, visualization_suggestions):
    """
    Suggest visualization types based on OpenAI's suggestions.
    
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        visualization_suggestions (str): Suggestions from OpenAI.
    
    Returns:
        list: List of suggested visualization types.
    """
    # Parse the visualization_suggestions string to extract visualization types
    # For simplicity, let's assume suggestions are comma-separated
    try:
        viz_list = [viz.strip().lower() for viz in visualization_suggestions.split(',')]
        logging.info(f"Visualization suggestions: {viz_list}")
        return viz_list
    except Exception as e:
        st.error(f"Failed to parse visualization suggestions: {e}")
        logging.error(f"Visualization suggestion parsing error: {e}")
        return []

def build_initial_model(df, target_column):
    """
    Build an initial machine learning model.
    
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        target_column (str): The column to predict.
    
    Returns:
        dict: Model evaluation metrics.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        logging.info("Machine learning model built successfully.")
        return {'mse': mse}
    except Exception as e:
        st.error(f"Failed to build machine learning model: {e}")
        logging.error(f"ML model error: {e}")
        return {}

def generate_narrative_insights(df, eda_results):
    """
    Generate narrative insights based on EDA results.
    
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        eda_results (dict): Results from EDA.
    
    Returns:
        str: Narrative insights.
    """
    # Implement narrative generation based on EDA results
    # This could involve summarizing the correlation matrix, highlighting key findings, etc.
    try:
        insights = "The dataset contains the following correlations among numeric variables:\n\n"
        corr = eda_results.get('correlation_matrix', pd.DataFrame())
        insights += corr.to_string()
        logging.info("Narrative insights generated successfully.")
        return insights
    except Exception as e:
        st.error(f"Failed to generate narrative insights: {e}")
        logging.error(f"Narrative insights error: {e}")
        return ""
