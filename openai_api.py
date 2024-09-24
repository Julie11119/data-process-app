# openai_api.py

import openai
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

def get_cleaning_suggestions(data_description):
    """
    Get data cleaning suggestions from OpenAI based on the dataset description.
    
    Args:
        data_description (str): Description of the dataset.
    
    Returns:
        str: Cleaning suggestions.
    """
    prompt = f"""
    You are a data scientist. Given the following dataset description, provide detailed suggestions for cleaning the data, including handling missing values, outliers, and data type corrections.

    Dataset Description:
    {data_description}

    Suggestions:
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful data scientist assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3,
        )
        # Extract the assistant's reply
        suggestions = response.choices[0].message['content'].strip()
        logging.info("Received cleaning suggestions from OpenAI.")
        return suggestions
    except openai.error.OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logging.error(f"OpenAI error: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error(f"Unexpected error: {e}")
        return ""

def get_visualization_suggestions(data_description):
    """
    Get visualization suggestions from OpenAI based on the dataset description.
    
    Args:
        data_description (str): Description of the dataset.
    
    Returns:
        str: Visualization suggestions.
    """
    prompt = f"""
    You are a data visualization expert. Given the following dataset description, suggest the most effective visualizations to uncover insights and trends.

    Dataset Description:
    {data_description}

    Visualization Suggestions:
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a helpful data visualization expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3,
        )
        # Extract the assistant's reply
        suggestions = response.choices[0].message['content'].strip()
        logging.info("Received visualization suggestions from OpenAI.")
        return suggestions
    except openai.error.OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logging.error(f"OpenAI error: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error(f"Unexpected error: {e}")
        return ""
