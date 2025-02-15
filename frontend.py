import streamlit as st
import requests
import json

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/generate"  # Update if running on a different host/port

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("📚 AI-Powered RAG Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "sources" not in st.session_state:
    st.session_state["sources"] = []

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
query = st.chat_input("Ask a question...")
if query:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Send request to FastAPI
    response = requests.post(API_URL, json={"question": query}, stream=True)
    
    # Read streaming response
    bot_reply = ""
    sources = []
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                text_chunk = chunk.decode("utf-8")
                bot_reply += text_chunk
                message_placeholder.markdown(bot_reply)
    
    # Extract sources from response headers or include it in response JSON
    try:
        sources = json.loads(response.headers.get("X-Sources", "[]"))
    except json.JSONDecodeError:
        sources = []

    # Save bot response and sources
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    st.session_state["sources"] = sources

# Dropdown for Sources
if st.session_state["sources"]:
    with st.expander("📖 Sources"):
        for i, source in enumerate(st.session_state["sources"], start=1):
            st.markdown(f"**Source {i}:**")
            st.text(f"📄 {source['source']} (Page {source['page']})")
            st.text_area("Excerpt:", source['page_content'], height=100, key=f"source_{i}")
