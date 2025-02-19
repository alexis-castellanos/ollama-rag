# ui.py
import time
import streamlit as st
from documents import process_document, add_to_vector_collection, delete_all_documents, delete_documents_by_source, get_distinct_sources
from analytics import render_analytics_dashboard, get_transcript
from retrieval import query_collection, cached_re_rank_cross_encoders
from llm_interface import call_llm

def render_sidebar(config, llm_config):
    """
    Render the sidebar with various sections:
      - Model Selection and Tuning
      - Upload Document
      - Manage Documents
      - Analytics Dashboard
      - Transcript Download
      - Document Source Filter
    Returns LLM tuning parameters and the selected document sources.
    """
    # --- Model Selection and Tuning ---
    with st.sidebar.expander("Model Selection and Tuning", expanded=True):
        llm_model = st.selectbox(
            "Select LLM Model",
            options=["deepseek-r1:latest", "llama3.2:latest", "llama3.2:1b"],
            index=2
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=2000, value=300, step=50)

    # --- Upload Document ---
    with st.sidebar.expander("Upload Document", expanded=True):
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)
        process = st.button("Process Document")
        if uploaded_file and process:
            if isinstance(uploaded_file, list):
                for file in uploaded_file:
                    original_file_name = file.name
                    normalized_file_name = original_file_name.translate(str.maketrans({"-": "_", ".": "_", ":": "_"}))
                    all_splits = process_document(file)
                    if all_splits:
                        add_to_vector_collection(all_splits, original_file_name, normalized_file_name)
            else:
                original_file_name = uploaded_file.name
                normalized_file_name = original_file_name.translate(str.maketrans({"-": "_", ".": "_", ":": "_"}))
                all_splits = process_document(uploaded_file)
                if all_splits:
                    add_to_vector_collection(all_splits, original_file_name, normalized_file_name)

    # --- Manage Documents ---
    with st.sidebar.expander("Manage Documents", expanded=True):
        if st.button("Clear All Documents"):
            delete_all_documents()
        distinct_sources = get_distinct_sources()
        if distinct_sources:
            search_query = st.text_input("Search Documents", "")
            sort_order = st.selectbox("Sort Order", options=["Ascending", "Descending"], index=0)
            filtered_sources = [source for source in distinct_sources if search_query.lower() in source.lower()]
            filtered_sources.sort(reverse=(sort_order == "Descending"))
            if filtered_sources:
                st.write("Or delete documents by file:")
                for source in filtered_sources:
                    if st.button(f"Delete {source}"):
                        delete_documents_by_source(source)
            else:
                st.write("No documents found matching search criteria.")

    # --- Analytics Dashboard ---
    with st.sidebar.expander("Show Analytics Dashboard"):
        render_analytics_dashboard()

    # --- Transcript Download ---
    transcript = get_transcript()
    st.sidebar.download_button(
        "Download Conversation Transcript",
        data=transcript,
        file_name="conversation.txt",
        mime="text/plain"
    )

    # --- Document Source Filter ---
    distinct_sources = get_distinct_sources()
    selected_sources = st.sidebar.multiselect("Filter by Document Source", options=distinct_sources, default=distinct_sources)

    # Return tuning parameters and the selected sources for query filtering
    return llm_model, temperature, max_tokens, selected_sources

def render_chat_ui(system_prompt, llm_model, temperature, max_tokens):
    """
    Render the main conversation/chat interface.
    Handles user input, querying, LLM calls, and displaying chat messages.
    """
    st.header("RAG Chat - Conversation")
    conversation_placeholder = st.empty()

    user_input = st.chat_input("Type your message here")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        start_time = time.time()
        with st.spinner("Querying the collection..."):
            # Use the document filter set in the sidebar (stored in session state)
            selected_sources = st.session_state.get("selected_sources", [])
            results = query_collection(user_input, sources=selected_sources)
        retrieved_docs = results.get("documents", [])
        retrieved_metadatas = results.get("metadatas", [])
        if retrieved_docs and retrieved_metadatas:
            docs = retrieved_docs[0]
            metadatas = retrieved_metadatas[0]
            if docs:
                relevant_text, _, citations, snippets = cached_re_rank_cross_encoders(user_input, docs, metadatas)
                st.session_state.last_context = relevant_text
                st.session_state.last_snippets = list(zip(snippets, citations))
                citations_str = "\n".join(citations)
                combined_prompt = f"{user_input}\n\nCitations:\n{citations_str}"
                complete_response = ""
                with st.spinner("Generating answer..."):
                    for chunk in call_llm(
                            context=relevant_text,
                            prompt=combined_prompt,
                            conversation_history=st.session_state.chat_history,
                            system_prompt=system_prompt,
                            llm_model=llm_model,
                            temperature=temperature,
                            max_tokens=max_tokens):
                        complete_response += chunk
                st.session_state.chat_history.append({"role": "assistant", "content": complete_response})
                end_time = time.time()
                from analytics import update_analytics
                update_analytics(citations, end_time - start_time)
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "No relevant documents found."})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "No relevant documents found."})

    # --- Render Chat Messages ---
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                cols = st.columns(3)
                if cols[0].button("👍", key=f"feedback_up_{i}"):
                    st.session_state.feedback[i] = "up"
                if cols[1].button("👎", key=f"feedback_down_{i}"):
                    st.session_state.feedback[i] = "down"
                feedback = st.session_state.feedback.get(i)
                if feedback:
                    st.markdown(f"Feedback: **{feedback}**")
                if st.session_state.feedback.get(i) == "down":
                    refine_text = cols[2].text_input("Refine answer", key=f"refine_{i}")
                    if refine_text:
                        st.session_state.chat_history.append({"role": "user", "content": refine_text})
                        refined_response = ""
                        with st.spinner("Generating refined answer..."):
                            for chunk in call_llm(
                                    context=st.session_state.last_context,
                                    prompt=refine_text,
                                    conversation_history=st.session_state.chat_history,
                                    system_prompt=system_prompt,
                                    llm_model=llm_model,
                                    temperature=temperature,
                                    max_tokens=max_tokens):
                                refined_response += chunk
                        st.session_state.chat_history.append({"role": "assistant", "content": refined_response})
                        st.session_state.feedback[i] = "refined"

    # --- Display Context Snippets ---
    if "last_snippets" in st.session_state and st.session_state.last_snippets:
        with st.expander("Context Snippets Used for Answering"):
            for snippet, citation in st.session_state.last_snippets:
                st.markdown(f"> {snippet}\n\n*Source: {citation}*")
