import os
import tempfile
import logging
import re
import time
import datetime

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama

# MUST be the first Streamlit command!
st.set_page_config(page_title="RAG Chat")

# Initialize session state keys if they don't exist.
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

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_system_prompt(file_path: str = "init_prompt.txt") -> str:
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

system_prompt = load_system_prompt()

@st.cache_resource(show_spinner=False)
def get_chroma_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma.db")
    collection = chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

def get_vector_collection() -> chromadb.Collection:
    return get_chroma_collection()

@st.cache_resource(show_spinner=False)
def load_cross_encoder() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n", "\n\n", "  ", "", "?", "!", "."],
        )
        return text_splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Error processing document: {e}")
        logging.error(f"Error processing document: {e}")
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def add_to_vector_collection(all_splits: list[Document], original_file_name: str, normalized_file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []
    upload_date = datetime.datetime.now().isoformat()
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadata = split.metadata.copy()
        metadata["source"] = original_file_name
        metadata["upload_date"] = upload_date
        metadatas.append(metadata)
        ids.append(f"{normalized_file_name}_{idx}")
    try:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        st.success("Data added to the collection")
    except Exception as e:
        st.error(f"Failed to add documents to collection: {e}")
        logging.error(f"Failed to add documents to collection: {e}")

def query_collection(prompt: str, n_results: int = 10, sources: list = None) -> dict:
    try:
        collection = get_vector_collection()
        if sources and len(sources) > 0:
            results = collection.query(query_texts=[prompt], n_results=n_results, where={"source": {"$in": sources}})
        else:
            results = collection.query(query_texts=[prompt], n_results=n_results)
        return results
    except Exception as e:
        st.error(f"Error querying vector collection: {e}")
        logging.error(f"Error querying vector collection: {e}")
        return {}

def highlight_query(text: str, query: str) -> str:
    words = query.split()
    for word in words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(lambda m: f"**{m.group(0)}**", text)
    return text

def re_rank_cross_encoders(prompt: str, documents: list[str], metadatas: list[dict]) -> tuple[str, list[int], list[str], list[str]]:
    encoder_model = load_cross_encoder()
    try:
        ranks = encoder_model.rank(prompt, documents, top_k=3)
    except Exception as e:
        st.error(f"Error in cross encoder ranking: {e}")
        logging.error(f"Error in cross encoder ranking: {e}")
        return "", [], [], []
    relevant_text = ""
    relevant_text_ids = []
    citations = []
    snippets = []
    for rank in ranks:
        idx = rank["corpus_id"]
        doc_text = documents[idx]
        relevant_text += doc_text + "\n\n"
        relevant_text_ids.append(idx)
        metadata = metadatas[idx]
        citation = f"{metadata.get('source', 'Unknown document')}"
        if "page" in metadata:
            citation += f", Page {metadata.get('page')}"
        citations.append(citation)
        snippet = doc_text[:300] + ("..." if len(doc_text) > 300 else "")
        highlighted_snippet = highlight_query(snippet, prompt)
        snippets.append(highlighted_snippet)
    return relevant_text, relevant_text_ids, citations, snippets

@st.cache_data(show_spinner=False, hash_funcs={dict: lambda d: tuple(sorted(d.items()))})
def cached_re_rank_cross_encoders(prompt: str, docs: list[str], metadatas: list[dict]):
    return re_rank_cross_encoders(prompt, docs, metadatas)

def call_llm(context: str, prompt: str, conversation_history: list[dict] = None,
             llm_model: str = "llama3.2:1b", temperature: float = 0.7, max_tokens: int = 300):
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\n{prompt}"})
    try:
        response = ollama.chat(
            model=llm_model,
            stream=True,
            messages=messages,
            options={"temperature": temperature, "max_tokens": max_tokens}
        )
        for chunk in response:
            if not chunk.get("done", True):
                yield chunk['message']['content']
            else:
                break
    except Exception as e:
        error_message = f"Error during LLM call: {e}"
        st.error(error_message)
        logging.error(error_message)
        yield error_message

def get_distinct_sources() -> list:
    collection = get_vector_collection()
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        distinct_sources = list({metadata.get("source", "Unknown") for metadata in metadatas})
        return distinct_sources
    except Exception as e:
        st.error(f"Error retrieving sources: {e}")
        logging.error(f"Error retrieving sources: {e}")
        return []

def delete_documents_by_source(source: str):
    collection = get_vector_collection()
    try:
        collection.delete(where={"source": source})
        st.success(f"Documents from '{source}' deleted successfully.")
    except Exception as e:
        st.error(f"Error deleting documents from '{source}': {e}")
        logging.error(f"Error deleting documents from '{source}': {e}")

def delete_all_documents():
    collection = get_vector_collection()
    try:
        collection.delete()
        st.success("All documents deleted successfully.")
    except Exception as e:
        st.error(f"Error deleting all documents: {e}")
        logging.error(f"Error deleting all documents: {e}")

def update_analytics(citations: list[str], response_time: float):
    st.session_state.analytics["query_count"] += 1
    st.session_state.analytics["total_response_time"] += response_time
    for citation in citations:
        st.session_state.analytics["citation_counts"][citation] = st.session_state.analytics["citation_counts"].get(citation, 0) + 1

def render_analytics_dashboard():
    analytics = st.session_state.analytics
    query_count = analytics["query_count"]
    avg_response_time = (analytics["total_response_time"] / query_count) if query_count else 0.0
    citation_counts = analytics["citation_counts"]
    st.subheader("Analytics Dashboard")
    st.markdown(f"**Total Queries:** {query_count}")
    st.markdown(f"**Average Response Time:** {avg_response_time:.2f} seconds")
    if citation_counts:
        st.markdown("**Citation Frequency:**")
        for citation, count in sorted(citation_counts.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"- {citation}: {count}")
    else:
        st.markdown("No citation data available.")

def get_transcript() -> str:
    transcript = ""
    for message in st.session_state.chat_history:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        transcript += f"{role}: {content}\n\n"
    return transcript

# ----------------------- Sidebar with Expandable Sections -----------------------

with st.sidebar.expander("Model Selection and Tuning", expanded=True):
    llm_model = st.selectbox(
        "Select LLM Model",
        options=["deepseek-r1:latest", "llama3.2:latest", "llama3.2:1b"],
        index=2
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)
    max_tokens = st.number_input("Max Tokens", min_value=50, max_value=2000, value=300, step=50)

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

with st.sidebar.expander("Show Analytics Dashboard"):
    render_analytics_dashboard()

# Multi-Document Filter for Queries
distinct_sources = get_distinct_sources()
selected_sources = st.sidebar.multiselect("Filter by Document Source", options=distinct_sources, default=distinct_sources)

# ----------------------- Export Conversation Transcript -----------------------

transcript = get_transcript()
st.sidebar.download_button(
    "Download Conversation Transcript",
    data=transcript,
    file_name="conversation.txt",
    mime="text/plain"
)

# ----------------------- Main Conversation UI -----------------------

st.header("RAG Chat - Conversation")
conversation_placeholder = st.empty()

user_input = st.chat_input("Type your message here")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    start_time = time.time()
    with st.spinner("Querying the collection..."):
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
                        llm_model=llm_model,
                        temperature=temperature,
                        max_tokens=max_tokens):
                    complete_response += chunk
            st.session_state.chat_history.append({"role": "assistant", "content": complete_response})
            end_time = time.time()
            update_analytics(citations, end_time - start_time)
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "No relevant documents found."})
    else:
        st.session_state.chat_history.append({"role": "assistant", "content": "No relevant documents found."})

with conversation_placeholder:
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
                                    llm_model=llm_model,
                                    temperature=temperature,
                                    max_tokens=max_tokens):
                                refined_response += chunk
                        st.session_state.chat_history.append({"role": "assistant", "content": refined_response})
                        st.session_state.feedback[i] = "refined"

if "last_snippets" in st.session_state and st.session_state.last_snippets:
    with st.expander("Context Snippets Used for Answering"):
        for snippet, citation in st.session_state.last_snippets:
            st.markdown(f"> {snippet}\n\n*Source: {citation}*")
