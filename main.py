# main.py
import streamlit as st
import time
import logging

# --- Set up Streamlit page configuration ---
st.set_page_config(page_title="RAG Chat")

# --- Initialize session state keys ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "query_count": 0,
        "total_response_time": 0.0,
        "citation_counts": {}
    }

# --- Load configuration and system prompt ---
from config import load_config, load_system_prompt
config = load_config("config.json")
system_prompt = load_system_prompt("init_prompt.txt")

# --- Render UI components ---
from ui import render_sidebar, render_chat_ui

# Render the sidebar and retrieve tuning parameters and document filter settings.
llm_model, temperature, max_tokens, selected_sources = render_sidebar(config, config.get("llm", {}))
st.session_state["selected_sources"] = selected_sources

# Render the main chat interface.
render_chat_ui(system_prompt, llm_model, temperature, max_tokens)
