# data_processing.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai_api import get_cleaning_suggestions, get_visualization_suggestions

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(uploaded_file, file_type):
    """
    Load data from various file types.
    
    Args:
        uploaded_file (UploadedFile): The uploaded file from Streamlit.
        file_type (str): Type of the file (e.g., 'csv', 'xlsx', 'json').
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        ValueError: If the file type is unsupported.
    """
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type in ["xlsx", "xls", "xlsm"]:
        df = pd.read_excel(uploaded_file)
    elif file_type == "json":
        df = pd.read_json(uploaded_file)
    else:
        raise ValueError("Unsupported file type.")
    return df

def generate_data_summary(df):
    """
    Generate a comprehensive summary of the dataset.
    
    Args:
        df (pd.DataFrame): The dataset.
    
    Returns:
        str: Text summary of the dataset.
    """
    summary = f"""
    Dataset has {df.shape[0]} rows and {df.shape[1]} columns.
    
    Columns and Data Types:
    {df.dtypes.to_string()}
    
    Missing Values:
    {df.isnull().sum().to_string()}
    
    Statistical Summary:
    {df.describe(include='all').to_string()}
    """
    return summary

def clean_data(df, cleaning_suggestions):
    """
    Clean data based on suggestions from OpenAI.
    
    Args:
        df (pd.DataFrame): Original dataset.
        cleaning_suggestions (str): Textual suggestions for cleaning.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # For simplicity, use keyword-based parsing. For more robust solutions, consider NLP parsing.
    
    # Remove duplicates if suggested
    if "remove duplicates" in cleaning_suggestions.lower():
        df = df.drop_duplicates()
    
    # Handle missing values based on suggestions
    if "fill missing values" in cleaning_suggestions.lower():
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert data types if suggested
    if "convert data types" in cleaning_suggestions.lower():
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass  # Keep as object if conversion fails
    
    # Additional cleaning steps can be implemented based on OpenAI's suggestions
    
    return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis and generate visualizations.
    
    Args:
        df (pd.DataFrame): Cleaned dataset.
    
    Returns:
        dict: Dictionary containing EDA results and figures.
    """
    eda_results = {}
    
    # Correlation Heatmap
    corr = df.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    eda_results['correlation_heatmap'] = fig_corr
    
    # Distribution of Numeric Features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        eda_results[f'distribution_{col}'] = fig
    
    # Categorical Features Count
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col,
                     labels={'index': col, col: 'Count'}, title=f"Count of {col}")
        eda_results[f'count_{col}'] = fig
    
    # Pairplot for Numeric Features
    if len(numeric_cols) >= 2:
        fig_pair = px.scatter_matrix(df, dimensions=numeric_cols, title="Pairplot of Numeric Features")
        eda_results['pairplot'] = fig_pair
    
    return eda_results

def suggest_visualizations(df, visualization_suggestions):
    """
    Suggest visualizations based on OpenAI's recommendations.
    
    Args:
        df (pd.DataFrame): Cleaned dataset.
        visualization_suggestions (str): Textual suggestions for visualizations.
    
    Returns:
        list: List of suggested visualization types.
    """
    # Simple keyword-based parsing
    suggestions = []
    if "histogram" in visualization_suggestions.lower():
        suggestions.append("Histogram")
    if "scatter plot" in visualization_suggestions.lower():
        suggestions.append("Scatter Plot")
    if "box plot" in visualization_suggestions.lower():
        suggestions.append("Box Plot")
    if "heatmap" in visualization_suggestions.lower():
        suggestions.append("Heatmap")
    if "bar chart" in visualization_suggestions.lower():
        suggestions.append("Bar Chart")
    if "pie chart" in visualization_suggestions.lower():
        suggestions.append("Pie Chart")
    if "pairplot" in visualization_suggestions.lower():
        suggestions.append("Pairplot")
    if "choropleth" in visualization_suggestions.lower():
        suggestions.append("Choropleth Map")
    
    return suggestions

def build_initial_model(df, target_column):
    """
    Build and evaluate a simple linear regression model.
    
    Args:
        df (pd.DataFrame): The dataset.
        target_column (str): The column to predict.
    
    Returns:
        dict: Model performance metrics and the trained model.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Handle any remaining missing values
    X = X.fillna(0)
    y = y.fillna(y.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return {
        'model': model,
        'mse': mse
    }

def generate_narrative_insights(df, eda_results):
    """
    Generate narrative insights based on EDA results using OpenAI.
    
    Args:
        df (pd.DataFrame): The dataset.
        eda_results (dict): Results from EDA functions.
    
    Returns:
        str: Narrative insights.
    """
    summary = generate_data_summary(df)
    prompt = f"""
    Based on the following dataset summary and EDA results, provide a comprehensive narrative insights report.

    Dataset Summary:
    {summary}

    EDA Results:
    {eda_results}

    Insights:
    """
    
    response = get_cleaning_suggestions(prompt)  # Reuse the cleaning_suggestions function for simplicity
    
    return response
