# data_processing.py

import pandas as pd
import numpy as np
import streamlit as st
import logging
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

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

@st.cache_data(show_spinner=False)
def clean_data(df, cleaning_suggestions):
    """
    Clean the DataFrame based on structured suggestions.

    Args:
        df (pd.DataFrame): The original DataFrame.
        cleaning_suggestions (dict): Structured cleaning suggestions.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        # Handle Missing Values
        mv = cleaning_suggestions.get("missing_values", {})
        mv_strategy = mv.get("strategy")
        mv_columns = mv.get("columns", [])
        mv_fill = mv.get("fill_value", None)
        
        if mv_strategy == "drop" and mv_columns:
            df = df.dropna(subset=mv_columns)
            logging.info(f"Dropped rows with missing values in columns: {mv_columns}")
        
        elif mv_strategy == "fill" and mv_columns:
            for col in mv_columns:
                if mv_fill == "mean":
                    if df[col].dtype in [np.int64, np.float64]:  # Check if numeric column
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        st.warning(f"⚠️ Column '{col}' is not numeric, cannot fill with mean.")
                    # fill_values[col] = df[col].mean()
                elif mv_fill == "median":
                    if df[col].dtype in [np.int64, np.float64]:  # Check if numeric column
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        st.warning(f"⚠️ Column '{col}' is not numeric, cannot fill with median.")
                
                # elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                #     mode_value = df[col].mode()[0]
                #     df[col] = df[col].fillna(mode_value)
                #     st.info(f"Filled missing values in column '{col}' with mode (most frequent value): {mode_value}.")

                else:
                # Ensure fill_value is the same type as the column
                    if isinstance(mv_fill, str):
                        df[col] = df[col].fillna(str(mv_fill))
                    elif isinstance(fill_value, (int, float)):
                        df[col] = df[col].fillna(float(mv_fill))
                    else:
                        df[col] = df[col].fillna(mv_fill)  # Handle as generic object type
                        
        else:
            fill_values[col] = df[col].mode()[0]  # Default to mode if unspecified
            # df = df.fillna(value=fill_values)
            logging.info(f"Filled missing values in columns: {mv_columns} with {mv_fill}")
        
        # Handle Outliers
        outliers = cleaning_suggestions.get("outliers", {})
        outlier_strategy = outliers.get("strategy")
        outlier_columns = outliers.get("columns", [])
        outlier_method = outliers.get("method")
        cap_value = outliers.get("cap_value")
        
        if outlier_strategy and outlier_columns:
            for col in outlier_columns:
                if outlier_strategy == "remove":
                    if outlier_method == "IQR":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        logging.info(f"Removed outliers in column {col} using IQR.")
                    elif outlier_method == "Z-score":
                        z_scores = np.abs(stats.zscore(df[col]))
                        df = df[z_scores < 3]
                        logging.info(f"Removed outliers in column {col} using Z-score.")
                
                elif outlier_strategy == "cap":
                    if outlier_method == "IQR":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df[col] = np.where(df[col] < lower_bound, lower_bound,
                                           np.where(df[col] > upper_bound, upper_bound, df[col]))
                        logging.info(f"Capped outliers in column {col} using IQR.")
                    elif outlier_method == "Z-score" and cap_value:
                        z_scores = stats.zscore(df[col])
                        df[col] = np.where(z_scores > cap_value, cap_value, df[col])
                        df[col] = np.where(z_scores < -cap_value, -cap_value, df[col])
                        logging.info(f"Capped outliers in column {col} using Z-score with cap value {cap_value}.")
        
        # Handle Data Types
        data_types = cleaning_suggestions.get("data_types", {})
        for col, dtype in data_types.items():
            try:
                if dtype == "int":
                    df[col] = df[col].astype(int)
                elif dtype == "float":
                    df[col] = df[col].astype(float)
                elif dtype == "category":
                    df[col] = df[col].astype('category')
                elif dtype == "datetime":
                    df[col] = pd.to_datetime(df[col])
                logging.info(f"Converted column {col} to {dtype}.")
            except Exception as e:
                logging.warning(f"Failed to convert column {col} to {dtype}: {e}")
        
        # Handle Additional Steps
        additional_steps = cleaning_suggestions.get("additional_steps", [])
        for step in additional_steps:
            # Implement additional cleaning steps as needed
            # Example: If step is "remove duplicate rows"
            if "remove duplicate rows" in step.lower():
                df = df.drop_duplicates()
                logging.info("Removed duplicate rows.")
            elif "normalize data" in step.lower():
                scaler = MinMaxScaler()
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                logging.info("Normalized numeric columns.")
            # Add more conditional steps based on expected additional steps
        
        logging.info("Data cleaning based on suggestions completed.")
        return df
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        st.error(f"⚠️ Data cleaning failed: {e}")
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
        
        # Additional EDA tasks can be added here (e.g., distributions, summary statistics)
        
        eda_results = {
            'correlation_matrix': corr,
            # Add other EDA results here
        }
        
        logging.info("EDA performed successfully.")
        return eda_results
    except Exception as e:
        st.error(f"⚠️ An error occurred during EDA: {e}")
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
        st.error(f"⚠️ Failed to parse visualization suggestions: {e}")
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
        
        # Identify numeric and categorical columns for encoding
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # One-Hot Encode categorical variables
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        logging.info("Machine learning model built successfully.")
        return {'mse': mse}
    except Exception as e:
        st.error(f"⚠️ Failed to build machine learning model: {e}")
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
    try:
        insights = "### Narrative Insights\n\n"
        corr = eda_results.get('correlation_matrix', pd.DataFrame())
        
        if corr.empty:
            insights += "No correlations to report."
            return insights
        
        # Find top 3 positive and negative correlations
        corr_unstacked = corr.abs().unstack()
        corr_unstacked = corr_unstacked[corr_unstacked < 1]  # Exclude self-correlation
        top_correlations = corr_unstacked.sort_values(ascending=False).drop_duplicates().head(6)
        
        insights += "The top correlations in the dataset are:\n\n"
        for idx, value in top_correlations.items():
            insights += f"- **{idx[0]}** and **{idx[1]}**: {value:.2f}\n"
        
        return insights
    except Exception as e:
        st.error(f"⚠️ Failed to generate narrative insights: {e}")
        logging.error(f"Narrative insights error: {e}")
        return ""

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

