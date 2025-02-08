# RAG Chatbot using FastAPI and Streamlit

This project is a Retrieval-Augmented Generation (RAG) chatbot built with FastAPI for the backend and Streamlit for the frontend. The chatbot allows users to upload documents, process them into a vector database, and ask questions related to the uploaded content.

## Features
- **File Upload & Processing**: Uploads text or PDF files and extracts their content.
- **Chunking & Vectorization**: Processes text into chunks and stores them in a vector index.
- **Chat with AI**: Uses a retriever and an LLM to generate context-aware responses.
- **Streaming Responses**: Real-time response streaming for better user experience.

## Tech Stack
- **Backend**: FastAPI, LangChain, OpenAI API, FAISS
- **Frontend**: Streamlit
- **Deployment**: Docker

## Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.12+
- Docker for containerized deployment

### 1. Clone the Repository
```sh
git clone https://github.com/MarwanMohamed95/AI-Chatbot-RAG.git
cd AI_Chatbot
```

### 2. Set Up a Virtual Environment
```sh
conda create -n myenv python=3.12
conda activate myenv
```

### 3. Install Dependencies
```sh
python3 -m pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file and configure the necessary settings:

### 5. Run the Backend
```sh
python3 app.py
```

### 6. Run the Frontend
```sh
python3 -m streamlit run fastapi_with_streamlit_app.py
```

## Usage
1. Upload a document via the Streamlit UI.
2. The backend processes and stores it in a vector database.
3. Ask questions related to the document.
4. Responses are streamed back in real time.

## API Documentation  

### 1. **Upload and Process File**  
**Endpoint:** `POST /api/v1/upload-and-process/`  
**Description:** Uploads a file, validates it, extracts text, cleans it, chunks it, and stores it in a vector index for retrieval.  

**Request:**
  - `file` (required) – The document file to be uploaded (`.txt`, `.pdf`).  

**Response:**  
- **200 OK:**  If the chunking and processing done successfully.
  

### 2. **Generate Answer**  
**Endpoint:** `POST /api/v1/generate-answer/`  
**Description:** Generates an AI-powered answer based on the uploaded document using retrieval-augmented generation (RAG).  

**Request:**  
  - `question` (required) – The user’s query based on the uploaded document.  

**Response (Streaming):**  
- **200 OK:** Returns a streamed response with tokens of the generated answer.  

## Deployment
### Docker
# Build the Docker image
docker build -t chatbot .

# Run the container
docker run -p 5000:5000 chatbot
