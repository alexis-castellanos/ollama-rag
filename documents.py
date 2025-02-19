# documents.py
import os
import tempfile
import datetime
import logging
import streamlit as st

import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama

@st.cache_resource(show_spinner=False)
def get_chroma_collection() -> chromadb.Collection:
    """
    Initialize and return the Chroma collection used for storing document embeddings.
    """
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
    """
    Return the vector collection by calling the cached get_chroma_collection.
    """
    return get_chroma_collection()

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """
    Process an uploaded PDF by saving it temporarily, loading its content,
    and splitting it into chunks.
    """
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
    """
    Add processed document chunks along with metadata (such as source and upload date)
    to the vector collection.
    """
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

def delete_documents_by_source(source: str):
    """
    Delete documents from the vector store by filtering on the document source.
    """
    collection = get_vector_collection()
    try:
        collection.delete(where={"source": source})
        st.success(f"Documents from '{source}' deleted successfully.")
    except Exception as e:
        st.error(f"Error deleting documents from '{source}': {e}")
        logging.error(f"Error deleting documents from '{source}': {e}")

def delete_all_documents():
    """
    Delete all documents from the vector store.
    """
    collection = get_vector_collection()
    try:
        collection.delete()
        st.success("All documents deleted successfully.")
    except Exception as e:
        st.error(f"Error deleting all documents: {e}")
        logging.error(f"Error deleting all documents: {e}")

def get_distinct_sources() -> list:
    """
    Retrieve a list of unique document sources stored in the vector collection.
    """
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
