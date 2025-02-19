# retrieval.py
import re
import logging
import streamlit as st

from sentence_transformers import CrossEncoder
from documents import get_vector_collection

@st.cache_resource(show_spinner=False)
def load_cross_encoder() -> CrossEncoder:
    """
    Load and cache the cross encoder model used for re-ranking search results.
    """
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def query_collection(prompt: str, n_results: int = 10, sources: list = None) -> dict:
    """
    Query the vector collection for relevant documents based on the prompt.
    Optionally filter by document sources.
    """
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
    """
    Highlight query words within the given text.
    """
    words = query.split()
    for word in words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(lambda m: f"**{m.group(0)}**", text)
    return text

def re_rank_cross_encoders(prompt: str, documents: list[str], metadatas: list[dict]) -> tuple[str, list[int], list[str], list[str]]:
    """
    Re-rank the retrieved documents using a cross encoder model.
    Returns a tuple with concatenated relevant text, document IDs, citations, and highlighted snippets.
    """
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
    """
    A cached version of the re-ranking function to speed up repeated queries.
    """
    return re_rank_cross_encoders(prompt, docs, metadatas)
