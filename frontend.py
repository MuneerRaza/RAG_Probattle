import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/generate"  # Update if running on a different host/port

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("📚 AI-Powered RAG Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                text_chunk = chunk.decode("utf-8")
                bot_reply += text_chunk
                message_placeholder.markdown(bot_reply)
    
    # Save bot response
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
