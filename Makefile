.PHONY: up down reset ingest test evaluate graph chat init

ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python
    PIP    := .venv/Scripts/pip
else
    PYTHON := .venv/bin/python
    PIP    := .venv/bin/pip
endif

## Initialize the project by installing dependencies
init:
	$(PYTHON) -m venv .venv
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
	@make ingest

## Stop Docker services only (keep volumes)
down:
	docker compose down

## Run all unit tests
test:
	$(PYTHON) -m pytest tests/ -v

## Run Admin API unit tests
test-admin:
	docker build --target test -t admin-api-test ./admin_api
	docker run --rm --env-file .env admin-api-test

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