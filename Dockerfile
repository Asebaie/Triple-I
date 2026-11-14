FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY rag.py app.py ./
COPY data/knowledge-base.zip data/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
