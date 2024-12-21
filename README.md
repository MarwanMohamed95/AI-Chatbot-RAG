# Retrieval-Augmented Generation (RAG) Assistant

## Overview
The **RAG Assistant** is an interactive AI-powered system built with Streamlit, LangChain, and Ollama. It integrates a conversational interface with Retrieval-Augmented Generation (RAG) capabilities, allowing users to upload documents, retrieve relevant context, and generate intelligent responses. The assistant leverages FAISS for efficient vector search and supports multiple file formats for input.

### Key Features
- **File Upload and Processing**: Supports `txt`, `pdf`, and `html` files, extracting and chunking text for context-aware retrieval.
- **Model Integration**: Uses Ollama models for summarization and question-answering tasks.
- **Context-Aware Retrieval**: Dynamically fetches relevant information from uploaded files to answer user queries.
- **Chat Interface**: Interactive chat experience with contextually relevant, token-by-token streamed responses.

---

## Setup Instructions

### Prerequisites
- Python 3.11
- Virtual environment tool (e.g., `conda`)

### Installation


1. **Create a Virtual Environment**:
   ```bash
   conda create --name rag_env python=3.11
   conda activate rag_env
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Additional Dependencies**:
   Ensure you have `FAISS` and `Ollama` installed for vector search and model handling.

---

## Running the Application

1. **Launch Streamlit**:
    
    Change directory to src folder and run the command below:
   ```bash
   python3 -m streamlit run src/app.py
   ```

2. **Access the Interface**:
   Open the link displayed in the terminal (typically `http://localhost:8501`) in your browser.

3. **Upload Files**:
   - Use the sidebar to upload files (`txt`, `pdf`, `html`).
   - Wait for the system to process and embed the content.

4. **Select Models**:
   - Choose a summarization model for context-aware query handling.
   - Select a question-answering model for generating responses.

5. **Start Chatting**:
   - Use the chat input to ask questions.
   - The assistant will retrieve context from the uploaded files and generate responses.

---

## File Structure

- **`app.py`**: Entry point for the Streamlit application.
- **`data_loader.py`**: Handles file reading and chunking.
- **`embedding_faiss.py`**: Creates FAISS embeddings and vector index.
- **`context_aware.py`**: Manages context-aware retrieval.
- **`text_generation.py`**: Builds and integrates question-answering chains.
- **`requirements.txt`**: Lists required Python packages.

