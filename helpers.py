# utils/helpers.py

import pandas as pd

def identify_key_columns(df):
    """
    Identify key columns in the DataFrame for analysis.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict: Dictionary categorizing columns as numeric, categorical, datetime, etc.
    """
    key_cols = {
        'numeric': df.select_dtypes(include=[pd.np.number]).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    return key_cols
