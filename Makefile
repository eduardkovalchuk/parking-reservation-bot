.PHONY: up down reset chat test evaluate graph

## Start all services
up:
	docker compose up -d

## Stop services (keep volumes)
down:
	docker compose down

## Wipe volumes and restart (data ingestion runs automatically on chatbot startup)
reset:
	docker compose down -v
	docker compose up -d

## Start chat in CLI
chat:
	docker compose exec -it chatbot python main.py  

## Run all unit tests inside the chatbot container
test:
	docker compose exec chatbot python -m pytest tests/ -v

evaluate:
	docker compose exec chatbot python scripts/evaluate.py

## Save LangGraph diagram to graph.png in the current directory
graph:
	docker compose exec chatbot python main.py --save-graph /tmp/graph.png
	docker compose cp chatbot:/tmp/graph.png ./graph.png