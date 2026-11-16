FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY core core
COPY web web
COPY data/knowledge-base.zip data/

ENV PYTHONPATH=/app/core:/app/web

EXPOSE 8501

CMD ["streamlit", "run", "web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
