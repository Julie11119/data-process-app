# openai_api.py

import openai
import streamlit as st

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

def get_cleaning_suggestions(data_description):
    """
    Uses OpenAI to provide data cleaning suggestions based on the data description.
    
    Args:
        data_description (str): Summary of the dataset.
    
    Returns:
        str: Suggestions for data cleaning.
    """
    prompt = f"""
    You are a data scientist. Given the following dataset description, provide detailed suggestions for cleaning the data, including handling missing values, outliers, and data type corrections.

    Dataset Description:
    {data_description}

    Suggestions:
    """
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].text.strip()

def get_visualization_suggestions(data_description):
    """
    Uses OpenAI to suggest appropriate visualizations based on the data description.
    
    Args:
        data_description (str): Summary of the dataset.
    
    Returns:
        str: Suggestions for data visualizations.
    """
    prompt = f"""
    You are a data visualization expert. Given the following dataset description, suggest the most effective visualizations to uncover insights and trends.

    Dataset Description:
    {data_description}

    Visualization Suggestions:
    """
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].text.strip()
