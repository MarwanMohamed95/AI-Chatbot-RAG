import os
import streamlit as st
import requests
import json

# Configuration
BACKEND_URL = "http://localhost:5000/api/v1"

# Ensure session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

def upload_file(uploaded_file):
    """Upload and process file via backend"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{BACKEND_URL}/upload-and-process/", files=files)

        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data.get('session_id')
            st.session_state.file_uploaded = True
            st.success(f"File processed: {data.get('file_name', 'Unknown file')}")
            st.info(f"Chunks inserted: {data.get('inserted_chunks', 0)}")
        else:
            st.error(f"Upload failed: {response.json().get('signal', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error uploading file: {e}")

def chat_with_llm(session_id, question):
    """Stream responses from the backend"""
    payload = {"question": question}

    try:
        with requests.post(
            f"{BACKEND_URL}/generate-answer/",
            data=payload,
            headers={"Accept": "text/event-stream"},
            stream=True
        ) as response:
            if response.status_code == 200:
                response_text = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line == '[DONE]':
                            break
                        response_text += line
                        yield response_text
            else:
                st.error(f"Error: {response.status_code}")
                yield "An error occurred while generating the response."

    except Exception as e:
        st.error(f"Error communicating with backend: {e}")
        yield "An error occurred while processing your request."

# Streamlit UI
st.set_page_config(page_title="AI Chatbot")
st.title("AI Chatbot")

# Sidebar for file upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a document", 
        type=['txt', 'pdf'],
        help="Upload a document to chat with"
    )

    if uploaded_file and "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = True
        with st.spinner("Processing file... Please wait."):
            upload_file(uploaded_file)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response handling
if question := st.chat_input("What is in your mind?"):
    if "session_id" not in st.session_state:
        st.error("Please upload and process a document first")
    else:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Handle assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for response_chunk in chat_with_llm(st.session_state.session_id, question):
                full_response = response_chunk
                message_placeholder.markdown(full_response + "▌")
            
            # Remove cursor and display final response
            message_placeholder.markdown(full_response)
            
            # Store the complete response
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            