# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CityPark is an AI-powered parking reservation chatbot using a multi-agent architecture with Human-In-The-Loop (HITL) approval. All services run via Docker Compose.

## Commands

### Start / Stop

```bash
make up          # Start all services (docker compose up -d)
make down        # Stop services, keep volumes
make reset       # Wipe all volumes and restart (re-ingests data)
```

Without Make:
```bash
docker compose up -d
docker compose down -v && docker compose up -d   # full reset
```

### Testing

```bash
make test        # Run tests inside chatbot container
# Equivalent:
docker compose exec chatbot python -m pytest tests/ -v

# Single test file:
docker compose exec chatbot python -m pytest tests/test_chatbot.py -v
```

### Other Dev Commands

```bash
make chat        # Open chatbot CLI (terminal-based, no HITL)
make evaluate    # Run RAG evaluation suite
make graph       # Render LangGraph diagram → graph.png
```

## Access Points

- **Chatbot UI:** http://localhost:8501
- **Admin UI:** http://localhost:8510 (credentials: `ADMIN_USERNAME` / `ADMIN_PASSWORD`, default admin/admin)
- **Admin API docs:** http://localhost:8510/docs

## Environment Setup

Copy `.env.template` to `.env` and fill in:
- `OPENAI_API_KEY` — required, no default
- `ADMIN_USERNAME` / `ADMIN_PASSWORD` — required for admin login

All other variables (Weaviate, PostgreSQL, OpenAI models, RAG parameters) have sensible Docker defaults. LangSmith tracing is optional via `LANGSMITH_API_KEY`.

## Architecture

### Multi-Agent LangGraph Pipeline

Two agents run inside a single LangGraph state machine sharing `AgentState` (`chatbot/src/chatbot/state.py`):

```
START → input_guardrail
           ├─ (blocked) → blocked_response → output_guardrail → END
           └─ (ok) → Agent 1 (chat_agent)
                        ├─ (no booking) → output_guardrail → END
                        └─ (booking_requested) → Agent 2 (booking_agent)
                                                    └─ interrupt() [waits for admin]
                                                    └─ (resumed) → Agent 1 → ...
```

**Agent 1 — Chat Agent** (`chatbot/src/chatbot/chat_agent.py`): ReAct agent that answers parking questions via RAG and collects reservation details (name, surname, car_number, start/end datetime, space_type). Signals handoff by setting `booking_requested=True` via the `submit_for_booking` tool.

**Agent 2 — Booking Agent** (`chatbot/src/chatbot/booking_agent.py`): Structured-output agent that finds an available space, creates a pending reservation in PostgreSQL, then calls `interrupt()` to pause the graph. The Streamlit UI polls for the admin decision and calls `graph.resume()` to continue.

### Key Files

| File | Role |
|------|------|
| `chatbot/src/chatbot/graph.py` | LangGraph assembly, `chat()` and `resume()` helpers |
| `chatbot/src/chatbot/state.py` | `AgentState` TypedDict — single source of truth for state |
| `chatbot/src/chatbot/tools.py` | Agent 1's 5 tools (RAG, draft CRUD, cost calc, submit) |
| `chatbot/src/rag/retriever.py` | Hybrid retriever — routes query to Weaviate vs. SQL |
| `chatbot/src/database/vector_store.py` | Weaviate (static parking info via OpenAI embeddings) |
| `chatbot/src/database/sql_store.py` | PostgreSQL (prices, availability, reservations) |
| `chatbot/src/guardrails/filters.py` | Presidio PII detection — blocks input, redacts output |
| `chatbot/streamlit_app.py` | Chatbot web UI + HITL polling loop (2-sec interval) |
| `admin_api/main.py` | FastAPI approve/reject endpoints + admin dashboard |
| `data/seeds/init_db.sql` | PostgreSQL schema and seed data |
| `data/static/parking_info.md` | RAG source document (ingested into Weaviate on startup) |

### Hybrid Retrieval

`retriever.py` classifies each query by keyword ("price", "availability", "hours", "EV", etc.) and fetches from:
- **Weaviate** — semantic search over `parking_info.md` (static: location, policies, amenities, FAQ)
- **PostgreSQL** — live queries for prices, availability counts, working hours

Results are merged into a single context passed to the RAG chain (`rag/chain.py`).

### HITL Persistence

PostgreSQL serves double duty: application data (reservations) and LangGraph checkpoint storage (`langgraph-checkpoint-postgres`). This means HITL state survives container restarts — an interrupted graph can be resumed after a reboot.

### Guardrails

Presidio (`en_core_web_lg` spaCy model, baked into the chatbot Docker image):
- **Input:** Blocks messages containing CREDIT_CARD, IBAN_CODE, US_SSN
- **Output:** Redacts CREDIT_CARD, IBAN_CODE, US_SSN, US_BANK_NUMBER, CRYPTO

### Data Ingestion

On chatbot container startup, `entrypoint.sh` runs `scripts/ingest_data.py`, which chunks `parking_info.md` and upserts it into Weaviate. Re-ingestion is safe (idempotent). To force a clean re-ingest, run `make reset`.
