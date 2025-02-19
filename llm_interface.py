# llm_interface.py
import streamlit as st
import ollama

def call_llm(context: str, prompt: str, conversation_history: list = None,
             system_prompt: str = "", llm_model: str = "llama3.2:1b",
             temperature: float = 0.7, max_tokens: int = 300):
    """
    Call the LLM using Ollama with the provided context, prompt, and conversation history.
    Yields streaming output from the LLM.
    """
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
        yield error_message
