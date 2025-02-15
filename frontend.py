import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/generate"

st.title("RAG-based Question Answering")
st.write("Enter your query below:")

query = st.text_input("Question:")

# Check if FastAPI server is running before making requests
@st.cache_data
def is_api_running():
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

if not is_api_running():
    st.error("⚠️ API server is not running. Please start FastAPI and try again.")
else:
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                with requests.post(API_URL, json={"question": query}, stream=True) as response:
                    if response.status_code == 200:
                        st.success("Answer:")
                        answer_placeholder = st.empty()
                        answer_text = ""

                        # Stream response and update UI dynamically
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                answer_text += chunk.decode("utf-8")
                                answer_placeholder.write(answer_text)  # Update in real-time
                    else:
                        st.error("Error fetching answer. Please check API.")
        else:
            st.warning("Please enter a question.")
