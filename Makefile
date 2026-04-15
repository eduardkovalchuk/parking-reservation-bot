.PHONY: up down reset ingest test evaluate graph chat init

ifeq ($(OS),Windows_NT)
    PYTHON      := .venv/Scripts/python
    PIP         := .venv/Scripts/pip
    VENV_CREATE := if not exist .venv python -m venv .venv
else
    PYTHON      := .venv/bin/python
    PIP         := .venv/bin/pip
    VENV_CREATE := [ -d ".venv" ] || python3 -m venv .venv
endif

## Initialize the project by installing dependencies
init:
	$(VENV_CREATE)
	$(PIP) install -r requirements.txt

	# Download spaCy model (required for Presidio PII detection)
	$(PYTHON) -m spacy download en_core_web_lg

## Ingest static parking data into Weaviate
ingest:
	$(PYTHON) scripts/ingest_data.py

## Start Docker services (Weaviate + PostgreSQL)
up:
	docker compose up -d

## Stop Docker services and remove volumes (full data wipe)
reset:
	docker compose down -v
	@make up
	@$(PYTHON) -c "import time; print('Waiting for services...'); time.sleep(5)"
	@make ingest

## Stop Docker services only (keep volumes)
down:
	docker compose down

## Run all unit tests
test:
	$(PYTHON) -m pytest tests/ -v

## Run the RAG evaluation suite
## Usage: make evaluate
##         make evaluate NO_LLM_JUDGE=1
##         make evaluate OUTPUT=evaluation_report.json
##         make evaluate NO_LLM_JUDGE=1 OUTPUT=evaluation_report.json
NO_LLM_JUDGE ?=
OUTPUT        ?=
evaluate:
	$(PYTHON) scripts/evaluate.py \
		$(if $(NO_LLM_JUDGE),--no-llm-judge,) \
		$(if $(OUTPUT),--output $(OUTPUT),)

## Save the LangGraph state machine diagram to graph.png
graph:
	$(PYTHON) main.py --save-graph graph.png

## Run the chatbot in interactive mode
chat:
	$(PYTHON) main.py