FROM python:3.12-slim

WORKDIR /AI-Chatbot-RAG

COPY ./requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y curl

COPY . .

# Expose FastAPI port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
