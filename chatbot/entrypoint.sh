#!/bin/bash
set -e

echo "Ingesting parking data into Weaviate..."
python scripts/ingest_data.py

echo "Starting Streamlit UI..."
exec python -m streamlit run streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501
