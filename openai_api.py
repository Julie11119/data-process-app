# openai_api.py

import openai
import streamlit as st
import json
import re
import time
import logging
from jsonschema import validate, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# Define JSON schema for cleaning suggestions
cleaning_schema = {
    "type": "object",
    "properties": {
        "missing_values": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": ["drop", "fill"]},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "fill_value": {"type": ["string", "number", "null"]}
            },
            "required": ["strategy", "columns"]
        },
        "outliers": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": ["remove", "cap"]},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "method": {"type": "string", "enum": ["IQR", "Z-score"]},
                "cap_value": {"type": ["number", "null"]}
            },
            "required": ["strategy", "columns", "method"]
        },
        "data_types": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": ["int", "float", "category", "datetime"]
            }
        },
        "additional_steps": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["missing_values", "outliers", "data_types"]
}

def extract_json_from_response(response_text):
    """
    Extract JSON content from a response string, removing code fences if present.

    Args:
        response_text (str): The raw response from OpenAI.

    Returns:
        str: The extracted JSON string.
    """
    # Regex to find JSON within code fences
    json_match = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    else:
        # If no code fencing, attempt to find JSON object
        json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)
    return None

def get_cleaning_suggestions(data_description, retries=3):
    """
    Get data cleaning suggestions from OpenAI based on the dataset description.
    Implements a retry mechanism in case of parsing failures.

    Args:
        data_description (str): Description of the dataset.
        retries (int): Number of retries in case of failure.

    Returns:
        dict: Structured cleaning suggestions.
    """
    prompt = f"""
    You are a data scientist. Given the following dataset description, provide detailed suggestions for cleaning the data, including handling missing values, outliers, and data type corrections.

    Dataset Description:
    {data_description}

    Please provide your suggestions in valid JSON format following this structure exactly. Do not include any additional explanations or text. Use proper JSON syntax.

    ```json
    {{
        "missing_values": {{
            "strategy": "drop" or "fill",
            "columns": ["column1", "column2"],
            "fill_value": "mean" or "median" or specific_value
        }},
        "outliers": {{
            "strategy": "remove" or "cap",
            "columns": ["column3", "column4"],
            "method": "IQR" or "Z-score",
            "cap_value": numeric_value  // Must be a number, e.g., 100 or 3.5
        }},
        "data_types": {{
            "column5": "int" or "float" or "category" or "datetime",
            "column6": "int" or "float" or "category" or "datetime",
            ...
        }},
        "additional_steps": [
            "step1",
            "step2",
            ...
        ]
    }}
    ```
    """

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful data scientist assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.3,
            )
            suggestions = response.choices[0].message['content'].strip()

            # Extract JSON from response
            json_str = extract_json_from_response(suggestions)
            if not json_str:
                raise ValueError("No JSON object found in the response.")

            # Parse JSON
            suggestions_json = json.loads(json_str)

            # Validate JSON schema with enhanced error handling
            try:
                validate(instance=suggestions_json, schema=cleaning_schema)
            except ValidationError as ve:
                # Handle specific validation errors
                if "cap_value" in ve.message and ("number" in ve.message or "null" in ve.message):
                    st.warning(f"⚠️ Validation warning for cap_value: {ve.message}. Setting cap_value to null.")
                    # Safely set cap_value to None if it's invalid
                    outliers = suggestions_json.get('outliers', {})
                    if 'cap_value' in outliers:
                        outliers['cap_value'] = None
                    suggestions_json['outliers'] = outliers
                else:
                    raise ve  # Re-raise any other validation errors

            logging.info("Received and validated cleaning suggestions from OpenAI.")
            return suggestions_json
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                logging.info("Retrying to get a valid JSON response from OpenAI...")
                time.sleep(1)  # Brief pause before retrying
                continue
            else:
                st.error("⚠️ Failed to parse cleaning suggestions after multiple attempts.")
                logging.error(f"Failed after {retries} attempts.")
                return {}
        except openai.error.OpenAIError as e:
            st.error(f"⚠️ OpenAI API Error: {e}")
            logging.error(f"OpenAI error: {e}")
            return {}
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
            logging.error(f"Unexpected error: {e}")
            return {}

def get_visualization_suggestions(data_description, retries=3):
    """
    Get visualization suggestions from OpenAI based on the dataset description.
    Implements a retry mechanism in case of parsing failures.

    Args:
        data_description (str): Description of the dataset.
        retries (int): Number of retries in case of failure.

    Returns:
        list: List of suggested visualization types.
    """
    prompt = f"""
    You are a data visualization expert. Given the following dataset description, suggest the most effective visualizations to uncover insights and trends.

    Dataset Description:
    {data_description}

    Please provide your suggestions as a comma-separated list of visualization types. For example:
    Histogram, Scatter Plot, Box Plot
    """

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful data visualization expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
            )
            suggestions = response.choices[0].message['content'].strip()

            # Split the suggestions into a list
            viz_list = [viz.strip() for viz in suggestions.split(',') if viz.strip()]
            logging.info("Received visualization suggestions from OpenAI.")
            return viz_list
        except openai.error.OpenAIError as e:
            st.error(f"⚠️ OpenAI API Error: {e}")
            logging.error(f"OpenAI error: {e}")
            return []
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
            logging.error(f"Unexpected error: {e}")
            return []

