#!/bin/bash
trap "kill 0" EXIT

if [ ! -f "knowledge_base.db" ]; then
    python -m scripts.ingest_data
fi

uvicorn api.main:app --host 127.0.0.1 --port 8000 &
streamlit run frontend/app.py --server.port 8501