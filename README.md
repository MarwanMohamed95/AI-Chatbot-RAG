# Retrieval-Augmented Generation (RAG) Assistant

## Overview
The **RAG Assistant** is an interactive AI-powered system built with Streamlit, FastAPI, LangChain, and Ollama. It integrates a conversational interface with Retrieval-Augmented Generation (RAG) capabilities, allowing users to upload documents, retrieve relevant context, and generate intelligent responses. The assistant leverages FAISS for efficient vector search and supports multiple file formats for input.

### Key Features
- **File Upload and Processing**: Supports `txt`, `pdf`, `csv` and `html` files, extracting and chunking text for context-aware retrieval.
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

1. **Launch The Application**:

    Change directory to src folder and run the command below:
   
      `For FastAPI App:`

   ```bash
   python3 app.py
   ```

   `For Streamlit App:`

   ```bash
   python3 -m streamlit run streamlit_app.py
   ```

2. **Access the Interface**:
   Open the link displayed in the terminal in your browser.

3. **Upload Files**:
   - Upload files (`txt`, `pdf`, `html`, '`csv`).
   - Wait for the system to process and embed the content.

4. **Select Models**:
   - For `FastAPI` App choose the summarization and question-answering from `.env` file
   - For `Streamlit` App, select the model from the dropdown menu.

5. **Start Chatting**:
   - Use the chat input to ask questions.
   - The assistant will retrieve context from the uploaded files and generate responses.

# Docker Commands

### Build Docker Image

To build the Docker image, run the following command in the root directory of the project `rag_task` (where the `Dockerfile` is located):

```bash
docker build -t rag:v1 .
```

### Run the Container

```bash
docker run -d --name rag_app -p 5000:5000 rag:v1
```

#### For Evaluation I used the similarity between The input query and the retrieved documents and the response

you can find it at `evaluation.ipynb` notebook.
