"""
Data ingestion script: load static parking documents into Weaviate.

Usage:
    python scripts/ingest_data.py

The script:
1. Loads the Markdown file from data/static/parking_info.md
2. Splits it into overlapping chunks with LangChain's text splitter
3. Assigns category metadata based on section headers
4. Ingests all chunks into the Weaviate vector store
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import weaviate
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import get_settings
from src.database.vector_store import (
    delete_all_documents,
    ensure_collection_exists,
    get_weaviate_client,
    ingest_documents,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "data" / "static"


# Mapping from section keywords to category labels
_SECTION_CATEGORY_MAP: list[tuple[tuple[str, ...], str]] = [
    (("Overview",), "general"),
    (("Location", "Direction"), "location"),
    (("Level", "Space Type"), "spaces"),
    (("Amenities", "Features"), "amenities"),
    (("Operating Hours", "Hours"), "hours"),
    (("Booking Process", "Reserve", "Reservation"), "booking"),
    (("Rules", "Policies", "Policy"), "policies"),
    (("Payment",), "payment"),
    (("Contact",), "contact"),
    (("FAQ", "Frequently Asked"), "faq"),
]


def _infer_category(header_path: str, content: str) -> str:
    """Infer a category label from section header text."""
    text = f"{header_path} {content[:200]}"
    for keywords, category in _SECTION_CATEGORY_MAP:
        if any(k.lower() in text.lower() for k in keywords):
            return category
    return "general"


def load_and_split_markdown(path: Path, settings) -> list[Document]:
    """
    Load a Markdown file and split it into chunks, preserving header metadata.

    Uses MarkdownHeaderTextSplitter to chunk by headers, then
    RecursiveCharacterTextSplitter to enforce maximum chunk size.
    """
    text = path.read_text(encoding="utf-8")
    source = path.name

    # Split by Markdown headers first
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(text)

    # Further split large sections into overlapping chunks
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_docs: list[Document] = []
    for doc in header_docs:
        header_path = " > ".join(
            filter(None, [doc.metadata.get("h1"), doc.metadata.get("h2"), doc.metadata.get("h3")])
        )
        category = _infer_category(header_path, doc.page_content)

        sub_docs = char_splitter.split_documents([doc])
        for sub in sub_docs:
            sub.metadata["source"] = source
            sub.metadata["category"] = category
            sub.metadata["header_path"] = header_path
            final_docs.append(sub)

    return final_docs


def main() -> None:
    settings = get_settings()

    client: weaviate.WeaviateClient = get_weaviate_client()
    try:
        logger.info("Clearing existing Weaviate data…")
        delete_all_documents(client)

        all_docs: list[Document] = []
        for md_file in sorted(STATIC_DIR.glob("*.md")):
            logger.info("Processing file: %s", md_file.name)
            docs = load_and_split_markdown(md_file, settings)
            all_docs.extend(docs)
            logger.info("  Chunks created: %d", len(docs))

        if not all_docs:
            logger.warning("No documents found in %s", STATIC_DIR)
            return

        logger.info("Ingesting %d total chunks into Weaviate…", len(all_docs))
        count = ingest_documents(all_docs, client)
        logger.info("✓ Successfully ingested %d document chunks.", count)

    finally:
        client.close()


if __name__ == "__main__":
    main()
