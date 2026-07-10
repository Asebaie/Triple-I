FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir httpx aiosqlite PyMuPDF streamlit sqlalchemy networkx matplotlib python-dotenv scipy

COPY . .

RUN chmod +x run.sh

EXPOSE 8000
EXPOSE 8501

CMD ["./run.sh"]