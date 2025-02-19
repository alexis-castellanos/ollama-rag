# analytics.py
import streamlit as st

def update_analytics(citations: list, response_time: float):
    """
    Update session analytics with each query:
      - Increment query count
      - Update total response time
      - Track citation frequencies
    """
    st.session_state.analytics["query_count"] += 1
    st.session_state.analytics["total_response_time"] += response_time
    for citation in citations:
        st.session_state.analytics["citation_counts"][citation] = st.session_state.analytics["citation_counts"].get(citation, 0) + 1

def render_analytics_dashboard():
    """
    Render an analytics dashboard to display query statistics.
    """
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
    """
    Generate and return a transcript of the conversation stored in session state.
    """
    transcript = ""
    for message in st.session_state.chat_history:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        transcript += f"{role}: {content}\n\n"
    return transcript
