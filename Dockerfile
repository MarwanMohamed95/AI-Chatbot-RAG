FROM python:3.12-slim

WORKDIR /AI-Chatbot-RAG

COPY ./AI-Chatbot-RAG/requirements.txt /AI-Chatbot-RAG/

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y curl

COPY ./AI-Chatbot-RAG /AI-Chatbot-RAG

# Expose FastAPI port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
