# CityPark Parking Reservation Chatbot — Stage 1

An intelligent parking reservation chatbot built with **LangChain**, **LangGraph**, and **Retrieval-Augmented Generation (RAG)**. The system uses **Weaviate** as a vector database for static information and **PostgreSQL** for dynamic data (prices, availability, reservations).

---

## Architecture

![image](data/static/graph.png)

---

### Hybrid Retrieval

| Data Type | Storage | Examples |
|-----------|---------|---------|
| **Static** | Weaviate (vector search) | Location, amenities, policies, FAQ, booking guide |
| **Dynamic** | PostgreSQL (SQL queries) | Prices, space availability, working hours, reservations |

---

## Project Structure

```
parking-reservation-bot/
├── main.py                    # CLI entry point
├── docker-compose.yml         # Weaviate + PostgreSQL
├── requirements.txt
├── .env.example
├── data/
│   ├── static/
│   │   └── parking_info.md    # Source document (loaded into Weaviate)
│   └── seeds/
│       └── init_db.sql        # PostgreSQL schema + seed data
├── src/
│   ├── config.py              # Pydantic-settings configuration
│   ├── database/
│   │   ├── vector_store.py    # Weaviate client & ingestion
│   │   └── sql_store.py       # PostgreSQL queries (prices, availability, reservations)
│   ├── rag/
│   │   ├── retriever.py       # Hybrid retriever (Weaviate + PostgreSQL)
│   │   └── chain.py           # LangChain RAG answer generation chain
│   ├── chatbot/
│   │   ├── state.py           # LangGraph state definition (TypedDict)
│   │   ├── nodes.py           # All graph node functions
│   │   └── graph.py           # Graph assembly & chat() helper
│   ├── guardrails/
│   │   └── filters.py         # PII detection (regex + Presidio) & output sanitisation
│   └── evaluation/
│       └── metrics.py         # Precision@K, Recall@K, latency, LLM-as-judge
├── scripts/
│   ├── ingest_data.py         # Load static data into Weaviate
│   └── evaluate.py            # Run RAG evaluation suite
└── tests/
    ├── conftest.py
    ├── test_vector_store.py
    ├── test_sql_store.py
    ├── test_retriever.py
    ├── test_guardrails.py
    ├── test_chatbot.py
    └── test_evaluation.py
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Make
- Docker & Docker Compose
- OpenAI API key

### 2. Clone & install dependencies

```bash
# Install project's dependencies
make init
```

### 3. Configure environment

```bash
# Copy the template and fill in your API key
cp .env.template .env
```

The `.env` already contains your `OPENAI_API_KEY`. All other values use sensible Docker defaults.

### 4. Start services and ingest static data

```bash
make up
```

Wait ~15 seconds for Weaviate and PostgreSQL to initialise.

### 5. Ingest static data

```bash
make ingest
```

This splits `data/static/parking_info.md` into chunks and loads them into Weaviate.

### 6. Run the chatbot

```bash
make chat
```

Optional: preserve conversation state across sessions with a fixed thread ID:
```bash
make chat THREAD_ID=my-session-001
```

---

## Running Tests

```bash
make test
```

Unit tests mock all external services (Weaviate, PostgreSQL, OpenAI) — no live services required.

---

## Running the Evaluation Suite

Requires live Docker services and ingested data:

```bash
make evaluate

# Skip LLM-as-judge (faster/cheaper)
make evaluate NO_LLM_JUDGE=1

# Save results to JSON
make evaluate OUTPUT=evaluation_report.json
```

### Metrics reported

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of retrieved docs that are relevant |
| **Recall@K** | Fraction of relevant docs that were retrieved |
| **Hit Rate@K** | Whether any relevant doc appears in top-K |
| **Faithfulness** | LLM judge: does the answer stay within context? |
| **Answer Relevance** | LLM judge: is the answer on-point? |
| **Latency p50/p95/p99** | End-to-end response time percentiles |

---

## Guard Rails

1. **Input guard rail**: Detects credit card numbers, SSNs, and IBANs before processing.
2. **Output guard rail**: Scans the generated response for accidental data leaks (other customers' PII, financial data). Uses:
   - Microsoft Presidio (NLP-based NER) for deeper analysis when the spaCy model is available

---

## Models Used

| Role | Model | Reason |
|------|-------|--------|
| Chat/reasoning | `gpt-4o` | Cost-effective, strong reasoning for MVP |
| Embeddings | `text-embedding-3-small` | Cheapest OpenAI embedding, good quality |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | LLM for chat and intent classification |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `WEAVIATE_HOST` | `localhost` | Weaviate host |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_DB` | `parking_db` | Database name |
| `POSTGRES_USER` | `parking_user` | DB username |
| `POSTGRES_PASSWORD` | `parking_password` | DB password |
| `RETRIEVAL_K` | `5` | Number of vector search results |
| `CHUNK_SIZE` | `500` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `50` | Chunk overlap (characters) |
