# config.py
import os
import json
import logging
import streamlit as st

def load_config(config_path: str = "config.json") -> dict:
    """
    Load the configuration from a JSON file.
    Returns a default configuration if the file is not found.
    """
    if not os.path.exists(config_path):
        st.warning(f"{config_path} not found. Using default configuration.")
        logging.warning(f"{config_path} not found. Using default configuration.")
        # Default configuration settings
        return {
            "features": {
                "document_upload": True,
                "web_search": False,
                "image_generation": False,
                "local_llm": True
            },
            "llm": {
                "model": "llama3.2:latest",
                "temperature": 0.7,
                "max_tokens": 300
            }
        }
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        logging.error(f"Failed to load configuration: {e}")
        return {}

def load_system_prompt(file_path: str = "init_prompt.txt") -> str:
    """
    Load the system prompt from a file.
    Falls back to a default prompt if the file is not found or an error occurs.
    """
    if not os.path.exists(file_path):
        fallback_prompt = (
            "You are an AI assistant tasked with providing detailed answers based solely on the provided context. "
            "Whenever you use information from the context, please cite the source by referring to the document file name "
            "and page number (if available). If you directly quote any text, enclose the quote in quotation marks along with the citation. "
            "Do not include any information that is not present in the context."
        )
        st.warning(f"{file_path} not found. Using fallback system prompt.")
        logging.warning(f"{file_path} not found. Using fallback system prompt.")
        return fallback_prompt
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        st.error(f"Failed to load system prompt from {file_path}: {e}")
        logging.error(f"Failed to load system prompt from {file_path}: {e}")
        return ""
