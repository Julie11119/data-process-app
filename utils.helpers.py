# utils/helpers.py

import pandas as pd

def identify_key_columns(df):
    """
    Identify key columns such as categorical, numerical, and date columns.
    
    Args:
        df (pd.DataFrame): The dataset.
    
    Returns:
        dict: Dictionary with lists of categorical, numerical, and date columns.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    return {
        'categorical': categorical_cols,
        'numerical': numerical_cols,
        'date': date_cols
    }
