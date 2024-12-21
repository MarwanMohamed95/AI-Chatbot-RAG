FROM python:3.11-slim

WORKDIR /rag_task

COPY ./rag_task/requirements.txt /rag_task/

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y curl

COPY ./rag_task /rag_task

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
