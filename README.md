# CityPark Parking Reservation Chatbot

An intelligent parking reservation chatbot built with **LangChain**, **LangGraph**, and **Retrieval-Augmented Generation (RAG)**. The system uses a **ReAct agent** that can answer questions about the parking facility and collect reservation data conversationally, backed by **Weaviate** for semantic search and **PostgreSQL** for live data.

---

## Architecture

![Graph](data/static/graph.png)

The system is made up of two cooperating agents inside a single LangGraph graph, plus a separate Admin API service:

### Agent 1 — Chat agent
Handles all user-facing interactions: answers parking questions via RAG and collects reservation data conversationally.

| Tool | Purpose |
|------|---------|
| `retrieve_parking_info` | Hybrid retrieval: semantic search (Weaviate) + live SQL data (prices, availability, hours) |
| `get_reservation_draft` | Read current reservation fields from state |
| `update_reservation_draft` | Persist collected fields into state as the user provides them |
| `calculate_reservation_cost` | Compute the estimated cost before the user confirms |
| `submit_for_booking` | Hand off to the booking agent once the user explicitly confirms |

### Agent 2 — Booking agent (Human-in-the-Loop)
A backend worker that runs after the user confirms. It creates a pending reservation in PostgreSQL, then **pauses the graph** (`interrupt()`) until an admin approves or rejects via the Admin UI. The result is returned to Agent 1 as a structured `BookingResult`, which composes the final user-facing reply.

```
User confirms → Agent 1 → Agent 2 → creates pending reservation
                                   → graph pauses (interrupt)
                                   ← admin approves / rejects
                                   → Agent 2 returns BookingResult
               Agent 1 ← reads BookingResult → replies to user
```

### Hybrid Retrieval

| Data type | Storage | Examples |
|-----------|---------|---------|
| **Static** | Weaviate (vector search) | Location, amenities, policies, FAQ, booking guide |
| **Dynamic** | PostgreSQL (SQL queries) | Prices, space availability, working hours |

---

## Project Structure

```
parking-reservation-bot/
├── docker-compose.yml           # All services: Weaviate + PostgreSQL + chatbot + admin_api
├── Makefile                     # Convenience commands (optional)
├── requirements.txt
├── .env.template
├── data/
│   ├── static/
│   │   └── parking_info.md      # Source document (loaded into Weaviate on startup)
│   └── seeds/
│       └── init_db.sql          # PostgreSQL schema + seed data (150 spaces)
├── admin_api/
│   ├── Dockerfile
│   ├── main.py                  # FastAPI app (approve / reject endpoints + Admin UI)
│   ├── database.py              # PostgreSQL queries for the admin service
│   ├── config.py                # Pydantic-settings (admin credentials, DB connection)
│   ├── requirements.txt
│   └── static/
│       └── index.html           # Single-page Admin UI
└── chatbot/
    ├── Dockerfile
    ├── entrypoint.sh            # Ingest data → start Streamlit
    ├── main.py                  # CLI entry point
    ├── streamlit_app.py         # Web UI (chat + HITL approval polling)
    ├── src/
    │   ├── config.py            # Pydantic-settings configuration
    │   ├── chatbot/
    │   │   ├── state.py         # AgentState (MessagesState + reservation_data + flags)
    │   │   ├── tools.py         # Agent 1 tools
    │   │   ├── chat_agent.py    # Agent 1 — Q&A + reservation collection
    │   │   ├── booking_agent.py # Agent 2 — HITL booking worker
    │   │   ├── nodes.py         # Guardrail node functions
    │   │   └── graph.py         # Graph assembly, chat() and resume_after_admin_decision()
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

### 3. Open the chatbot UI

```
http://localhost:8501
```

### 4. Open the Admin UI

Reservations submitted by the chatbot arrive with status **pending** and must be approved or rejected by an admin before the user receives a confirmation.

```
http://localhost:8510
```

Login with the credentials set in `.env` (`ADMIN_USERNAME` / `ADMIN_PASSWORD`), default creads are **admin**/**admin** from `.env.template`.
The Admin UI lists all reservations and lets you approve or reject pending ones with a single click. The chatbot automatically detects the decision and resumes the conversation.

The Admin API also exposes a REST interface and interactive docs:

```
http://localhost:8510/docs
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
| `WEAVIATE_HOST` | `weaviate` | Weaviate service name (Docker internal) |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `WEAVIATE_COLLECTION_NAME` | `ParkingInfo` | Weaviate collection |
| `POSTGRES_HOST` | `postgres` | PostgreSQL service name (Docker internal) |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `parking_db` | Database name |
| `POSTGRES_USER` | `postgres` | DB username |
| `POSTGRES_PASSWORD` | `postgres` | DB password |
| `ADMIN_API_PORT` | `8510` | Port for the Admin API / Admin UI |
| `ADMIN_USERNAME` | — | **Required** — admin login username |
| `ADMIN_PASSWORD` | — | **Required** — admin login password |
| `RETRIEVAL_K` | `4` | Number of vector search results |
| `CHUNK_SIZE` | `450` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `70` | Chunk overlap (characters) |
