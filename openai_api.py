# openai_api.py

import openai
import streamlit as st
import logging
import json
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

def get_cleaning_suggestions(data_description):
    """
    Get data cleaning suggestions from OpenAI based on the dataset description.

    Args:
        data_description (str): Description of the dataset.

    Returns:
        dict: Structured cleaning suggestions.
    """
    prompt = f"""
    You are a data scientist. Given the following dataset description, provide detailed suggestions for cleaning the data, including handling missing values, outliers, and data type corrections.

    Dataset Description:
    {data_description}

    Please provide your suggestions in valid JSON format following this structure:

    {{
        "missing_values": {{
            "strategy": "drop" | "fill",
            "columns": ["column1", "column2"],
            "fill_value": "mean" | "median" | specific_value
        }},
        "outliers": {{
            "strategy": "remove" | "cap",
            "columns": ["column3", "column4"],
            "method": "IQR" | "Z-score",
            "cap_value": "upper_limit" | "lower_limit"  // Only if strategy is "cap"
        }},
        "data_types": {{
            "column5": "int" | "float" | "category" | "datetime",
            "column6": "int" | "float" | "category" | "datetime",
            ...
        }},
        "additional_steps": [
            "step1",
            "step2",
            ...
        ]
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Ensure you have access to GPT-4
            messages=[
                {"role": "system", "content": "You are a helpful data scientist assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
        )
        suggestions = response.choices[0].message['content'].strip()

        # Parse JSON
        suggestions_json = json.loads(suggestions)

        # Validate JSON schema with enhanced error handling
        try:
            validate(instance=suggestions_json, schema=cleaning_schema)
        except ValidationError as ve:
            # Check if the specific error is related to `cap_value` not being numeric
            if "cap_value" in ve.message and "number" in ve.message:
                st.warning(f"⚠️ Validation warning for cap_value: {ve.message}. Ignoring non-numeric values.")
                # Set cap_value to None or some default value
                for outlier_suggestion in suggestions_json.get('outliers', {}).get('columns', []):
                    outlier_suggestion['cap_value'] = None  # Set to None or default value
            else:
                raise ve  # Re-raise any other validation errors

        logging.info("Received and validated cleaning suggestions from OpenAI.")
        return suggestions_json
    except json.JSONDecodeError as e:
        st.error("⚠️ Failed to parse cleaning suggestions. Please ensure the response is in valid JSON format.")
        logging.error(f"JSON parsing error: {e}")
        return {}
    except ValidationError as ve:
        st.error(f"⚠️ Cleaning suggestions validation error: {ve.message}")
        logging.error(f"Validation error: {ve}")
        return {}
    except openai.error.OpenAIError as e:
        st.error(f"⚠️ OpenAI API Error: {e}")
        logging.error(f"OpenAI error: {e}")
        return {}
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")
        logging.error(f"Unexpected error: {e}")
        return {}
