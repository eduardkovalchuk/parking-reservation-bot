# CityPark Parking Reservation Chatbot

An intelligent parking reservation chatbot built with **LangChain**, **LangGraph**, and **Retrieval-Augmented Generation (RAG)**. The system uses a **ReAct agent** that can answer questions about the parking facility and collect reservation data conversationally, backed by **Weaviate** for semantic search and **PostgreSQL** for live data.

---

## Architecture

![Graph](data/static/graph.png)

The **agent** is a LangGraph ReAct agent with three tools:

| Tool | Purpose |
|------|---------|
| `retrieve_parking_info` | Hybrid retrieval: semantic search (Weaviate) + live SQL data (prices, availability, hours) |
| `get_reservation_draft` | Read current reservation fields from state |
| `update_reservation_draft` | Persist collected fields into state as the user provides them |

### Hybrid Retrieval

| Data type | Storage | Examples |
|-----------|---------|---------|
| **Static** | Weaviate (vector search) | Location, amenities, policies, FAQ, booking guide |
| **Dynamic** | PostgreSQL (SQL queries) | Prices, space availability, working hours |

---

## Project Structure

```
parking-reservation-bot/
├── docker-compose.yml           # All services: Weaviate + PostgreSQL + chatbot
├── Makefile                     # Convenience commands (optional)
├── requirements.txt
├── .env.template
├── data/
│   ├── static/
│   │   └── parking_info.md      # Source document (loaded into Weaviate on startup)
│   └── seeds/
│       └── init_db.sql          # PostgreSQL schema + seed data (150 spaces)
└── chatbot/
    ├── Dockerfile
    ├── entrypoint.sh            # Ingest data → start Streamlit
    ├── main.py                  # CLI entry point
    ├── streamlit_app.py         # Web UI
    ├── src/
    │   ├── config.py            # Pydantic-settings configuration
    │   ├── chatbot/
    │   │   ├── state.py         # AgentState (MessagesState + reservation_data + guardrail flags)
    │   │   ├── tools.py         # ReAct agent tools
    │   │   ├── agent.py         # create_agent() setup with system prompt
    │   │   ├── nodes.py         # Guardrail node functions
    │   │   └── graph.py         # Outer graph assembly & chat() helper
    │   ├── database/
    │   │   ├── vector_store.py  # Weaviate client & ingestion
    │   │   └── sql_store.py     # PostgreSQL queries (prices, availability, reservations)
    │   ├── rag/
    │   │   ├── retriever.py     # Hybrid retriever (Weaviate + PostgreSQL)
    │   │   └── chain.py         # RAG answer generation chain
    │   ├── guardrails/
    │   │   └── filters.py       # Presidio-based PII detection & output sanitisation
    │   └── evaluation/
    │       └── metrics.py       # Precision@K, Recall@K, latency, LLM-as-judge
    ├── scripts/
    │   ├── ingest_data.py       # Load static data into Weaviate
    │   └── evaluate.py          # RAG evaluation suite
    └── tests/
        ├── conftest.py
        ├── test_chatbot.py
        ├── test_guardrails.py
        ├── test_retriever.py
        ├── test_vector_store.py
        ├── test_sql_store.py
        └── test_evaluation.py
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key
- _(optional)_ Make

### 1. Configure environment

```bash
cp .env.template .env
# Fill in OPENAI_API_KEY and optionally LANGSMITH_API_KEY
```

### 2. Start all services

```bash
docker compose up -d
```

This starts Weaviate, PostgreSQL, and the chatbot. On first boot the chatbot container automatically ingests `parking_info.md` into Weaviate before launching the UI.

### 3. Open the web UI

```
http://localhost:8501
```

### Full reset (wipe all data and restart)

```bash
docker compose down -v
docker compose up -d
```

---

## Make Commands _(optional)_

If you have Make installed, the following shortcuts are available:

| Command | Description |
|---------|-------------|
| `make up` | Start all services |
| `make down` | Stop services (keep data) |
| `make reset` | Wipe volumes and restart |
| `make chat` | Start chatbot CLI inside the container |
| `make test` | Run unit tests inside the container |
| `make evaluate` | Run RAG evaluation suite inside the container |
| `make graph` | Render LangGraph diagram to `./graph.png` |

---

## Running Tests

```bash
# With Make
make test

# Without Make
docker compose exec chatbot python -m pytest tests/ -v
```

---

## Running the Evaluation Suite

Requires services to be running:

```bash
# With Make
make evaluate

# Without Make
docker compose exec chatbot python scripts/evaluate.py
```

---

## Guardrails

| Layer | What it checks | Action on detection |
|-------|---------------|-------------------|
| **Input** | Credit cards, SSNs, IBANs in user message | Block message, return safe refusal |
| **Output** | Sensitive financial data in generated reply | Replace reply with privacy notice |

Powered by [Microsoft Presidio](https://github.com/microsoft/presidio) with `en_core_web_lg` spaCy model.

---

## LangSmith Tracing

Set the following in `.env` to enable tracing:

```env
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT=parking-reservation-bot
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | LLM for agent reasoning |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for Weaviate |
| `WEAVIATE_HOST` | `localhost` | Weaviate host (auto-overridden to `weaviate` in Docker) |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `WEAVIATE_COLLECTION_NAME` | `ParkingInfo` | Weaviate collection |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host (auto-overridden to `postgres` in Docker) |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `parking_db` | Database name |
| `POSTGRES_USER` | `postgres` | DB username |
| `POSTGRES_PASSWORD` | `postgres` | DB password |
| `RETRIEVAL_K` | `4` | Number of vector search results |
| `CHUNK_SIZE` | `450` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `70` | Chunk overlap (characters) |
