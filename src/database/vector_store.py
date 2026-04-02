"""
Weaviate vector store integration for static parking information.

Static data stored here:
- General parking overview and description
- Location and directions
- Amenities and features
- Parking rules and policies
- Booking process guide
- FAQ
"""
from __future__ import annotations

import logging
from typing import List, Optional

import weaviate
import weaviate.classes as wvc
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

from src.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ParkingInfo"


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Create and return a Weaviate client connected to the local instance."""
    settings = get_settings()
    client = weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )
    return client


def ensure_collection_exists(client: weaviate.WeaviateClient) -> None:
    """Create the ParkingInfo collection if it does not already exist."""
    if client.collections.exists(COLLECTION_NAME):
        logger.info("Weaviate collection '%s' already exists.", COLLECTION_NAME)
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(
                name="content",
                data_type=wvc.config.DataType.TEXT,
                description="The text content of the document chunk.",
            ),
            wvc.config.Property(
                name="source",
                data_type=wvc.config.DataType.TEXT,
                description="Source file or section identifier.",
            ),
            wvc.config.Property(
                name="category",
                data_type=wvc.config.DataType.TEXT,
                description="Category: general, location, pricing, policies, faq, etc.",
            ),
        ],
    )
    logger.info("Created Weaviate collection '%s'.", COLLECTION_NAME)


def get_vector_store(client: weaviate.WeaviateClient) -> WeaviateVectorStore:
    """Return a LangChain WeaviateVectorStore backed by OpenAI embeddings."""
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    vector_store = WeaviateVectorStore(
        client=client,
        index_name=COLLECTION_NAME,
        text_key="content",
        embedding=embeddings,
        attributes=["source", "category"],
    )
    return vector_store


def ingest_documents(
    documents: List[Document],
    client: weaviate.WeaviateClient,
) -> int:
    """
    Ingest a list of LangChain Documents into Weaviate.

    Each Document should have metadata keys: 'source' and 'category'.
    Returns the number of documents ingested.
    """
    ensure_collection_exists(client)
    store = get_vector_store(client)
    ids = store.add_documents(documents)
    logger.info("Ingested %d document chunks into Weaviate.", len(ids))
    return len(ids)


def similarity_search(
    query: str,
    client: weaviate.WeaviateClient,
    k: int = 5,
    filter_category: Optional[str] = None,
) -> List[Document]:
    """
    Perform a semantic similarity search against the vector store.

    Args:
        query: The user's natural-language query.
        client: Active Weaviate client.
        k: Number of results to return.
        filter_category: Optional category to restrict results.

    Returns:
        List of the top-k most relevant Document objects.
    """
    store = get_vector_store(client)

    search_kwargs: dict = {"k": k}
    if filter_category:
        search_kwargs["where_filter"] = {
            "path": ["category"],
            "operator": "Equal",
            "valueText": filter_category,
        }

    results = store.similarity_search(query, **search_kwargs)
    return results


def delete_all_documents(client: weaviate.WeaviateClient) -> None:
    """Delete all documents from the ParkingInfo collection (useful for re-ingestion)."""
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        logger.info("Deleted Weaviate collection '%s'.", COLLECTION_NAME)
    ensure_collection_exists(client)
