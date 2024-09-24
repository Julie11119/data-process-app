# openai_api.py

import openai
import streamlit as st

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

def get_cleaning_suggestions(data_description):
    prompt = f"""
    You are a data scientist. Given the following dataset description, provide detailed suggestions for cleaning the data, including handling missing values, outliers, and data type corrections.

    Dataset Description:
    {data_description}

    Suggestions:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful data scientist assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # Extract the assistant's reply
    return response.choices[0].message['content'].strip()

def get_visualization_suggestions(data_description):
    prompt = f"""
    You are a data visualization expert. Given the following dataset description, suggest the most effective visualizations to uncover insights and trends.

    Dataset Description:
    {data_description}

    Visualization Suggestions:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful data visualization expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # Extract the assistant's reply
    return response.choices[0].message['content'].strip()
