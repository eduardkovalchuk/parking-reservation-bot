FROM python:3.13-slim

WORKDIR /app

# System deps for psycopg2 and spaCy native extensions (needed on ARM)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Build context is project root so requirements.txt is at the top level
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model — baked into image (~150 MB, cached by Docker layer)
RUN python -m spacy download en_core_web_lg

# Copy application source
COPY . .

RUN chmod +x entrypoint.sh \
    && sed -i 's/\r//' entrypoint.sh

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
