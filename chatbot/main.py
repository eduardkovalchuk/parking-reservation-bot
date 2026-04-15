"""
CityPark Parking Reservation Chatbot - CLI Entry Point

Usage:
    python main.py

Requires:
    - Docker services running: docker compose up -d
    - Data ingested: python scripts/ingest_data.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env into os.environ BEFORE any LangChain/LangSmith imports
# so that tracing env vars (LANGSMITH_API_KEY, etc.) are picked up.
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

import psycopg

from src.chatbot.graph import build_graph, chat, resume_after_admin_decision, ChatResult
from src.config import get_settings
from src.database import sql_store
from src.database.vector_store import get_weaviate_client
from unittest.mock import MagicMock

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


def _save_graph_image(output_path: str) -> None:
    """Render the LangGraph topology to a PNG file without live services."""
    mock_client = MagicMock()
    app = build_graph(mock_client)
    png_bytes: bytes = app.get_graph().draw_mermaid_png()
    path = Path(output_path)
    path.write_bytes(png_bytes)
    console.print(f"[green]✓ Graph saved to {path.resolve()}[/green]")


def print_welcome() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]🚗 CityPark Premium Parking Assistant[/bold blue]\n"
            "[dim]Powered by LangGraph + RAG | Type [bold]exit[/bold] or [bold]quit[/bold] to stop[/dim]",
            border_style="blue",
        )
    )
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="CityPark Parking Chatbot CLI")
    parser.add_argument(
        "--save-graph",
        type=str,
        metavar="FILE",
        default=None,
        help="Render the LangGraph state machine to a PNG file and exit (e.g. graph.png).",
    )
    args = parser.parse_args()

    if args.save_graph:
        _save_graph_image(args.save_graph)
        sys.exit(0)

    settings = get_settings()
    print_welcome()

    weaviate_client = None
    postgres_conn = None

    console.print("[dim]Connecting to services…[/dim]")
    try:
        weaviate_client = get_weaviate_client()
        weaviate_client.is_ready()  # Verify connection
    except Exception as exc:
        console.print(
            f"[bold red]✗ Could not connect to Weaviate:[/bold red] {exc}\n"
            "Make sure Docker services are running:  [bold]docker compose up -d[/bold]"
        )
        sys.exit(1)

    try:
        conn_string = (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        from langgraph.checkpoint.postgres import PostgresSaver
        postgres_conn = psycopg.connect(conn_string, autocommit=True)
        checkpointer = PostgresSaver(postgres_conn)
        checkpointer.setup()
    except Exception as exc:
        console.print(f"[bold red]✗ Failed to connect to PostgreSQL:[/bold red] {exc}")
        weaviate_client.close()
        sys.exit(1)

    try:
        app = build_graph(weaviate_client, checkpointer)
        console.print("[green]✓ Connected. Services ready.[/green]")
        console.print(Rule(style="dim"))
    except Exception as exc:
        console.print(f"[bold red]✗ Failed to build chatbot graph:[/bold red] {exc}")
        weaviate_client.close()
        postgres_conn.close()
        sys.exit(1)

    console.print()
    console.print("[bold]How can I help you today?[/bold]")
    console.print("[dim]You can ask about prices, location, availability, or book a parking space.[/dim]\n")

    try:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
                console.print("\n[blue]Thank you for using CityPark! Have a great day! 🚗[/blue]")
                break

            console.print()
            with console.status("[dim]Thinking…[/dim]", spinner="dots"):
                try:
                    result: ChatResult = chat(app, user_input, thread_id="cli-session")
                except Exception as exc:
                    logger.error("Chat error: %s", exc, exc_info=True)
                    result = ChatResult(
                        reply=(
                            "I'm sorry, something went wrong on my end. "
                            "Please try again or contact info@citypark.com or +31 20 555 0123."
                        )
                    )

            console.print(
                Panel(
                    Markdown(result.reply),
                    title="[bold green]Assistant[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                )
            )

            if result.interrupted:
                reservation_id = result.interrupt_reservation_id
                console.print(
                    f"\n[yellow]⏳ Reservation #{reservation_id} is pending admin approval.\n"
                    "Waiting… (approve or reject in the Admin UI at http://localhost:8510)[/yellow]"
                )
                while True:
                    time.sleep(4)
                    row = sql_store.get_reservation_by_id(reservation_id)
                    status = row["status"] if row else None
                    if status and status != "pending":
                        with console.status("[dim]Admin decided — finalising…[/dim]", spinner="dots"):
                            result = resume_after_admin_decision(
                                app, status, reservation_id, thread_id="cli-session"
                            )
                        console.print(
                            Panel(
                                Markdown(result.reply),
                                title="[bold green]Assistant[/bold green]",
                                border_style="green",
                                padding=(0, 1),
                            )
                        )
                        break

            console.print()

    finally:
        if weaviate_client:
            weaviate_client.close()
        if postgres_conn:
            postgres_conn.close()


if __name__ == "__main__":
    main()
