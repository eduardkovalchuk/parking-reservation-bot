"""
Tests for src/database/vector_store.py

These tests mock the Weaviate client and verify the module's logic
without requiring a running Weaviate instance.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import src.database.vector_store as vs


class TestEnsureCollectionExists:
    """Tests for ensure_collection_exists()."""

    def test_skips_creation_if_collection_already_exists(self, mock_weaviate_client):
        """If the collection already exists, no create call should be made."""
        mock_weaviate_client.collections.exists.return_value = True
        vs.ensure_collection_exists(mock_weaviate_client)
        mock_weaviate_client.collections.create.assert_not_called()

    def test_creates_collection_when_missing(self, mock_weaviate_client):
        """If the collection does not exist, it should be created."""
        mock_weaviate_client.collections.exists.return_value = False
        vs.ensure_collection_exists(mock_weaviate_client)
        mock_weaviate_client.collections.create.assert_called_once()

    def test_created_collection_has_correct_name(self, mock_weaviate_client):
        """The collection created should use the configured name."""
        mock_weaviate_client.collections.exists.return_value = False
        vs.ensure_collection_exists(mock_weaviate_client)
        call_kwargs = mock_weaviate_client.collections.create.call_args
        assert call_kwargs[1]["name"] == vs.COLLECTION_NAME or call_kwargs[0][0] == vs.COLLECTION_NAME


class TestIngestDocuments:
    """Tests for ingest_documents()."""

    def test_ingest_calls_add_documents(self, mock_weaviate_client):
        """ingest_documents should call store.add_documents and return count."""
        docs = [
            Document(page_content="Parking is great", metadata={"source": "test", "category": "general"}),
            Document(page_content="EV charging available", metadata={"source": "test", "category": "amenities"}),
        ]
        mock_store = MagicMock()
        mock_store.add_documents.return_value = ["id1", "id2"]

        with patch("src.database.vector_store.get_vector_store", return_value=mock_store), \
             patch("src.database.vector_store.ensure_collection_exists"):
            result = vs.ingest_documents(docs, mock_weaviate_client)

        assert result == 2
        mock_store.add_documents.assert_called_once_with(docs)

    def test_ingest_empty_list_returns_zero(self, mock_weaviate_client):
        """Ingesting an empty list should return 0."""
        mock_store = MagicMock()
        mock_store.add_documents.return_value = []

        with patch("src.database.vector_store.get_vector_store", return_value=mock_store), \
             patch("src.database.vector_store.ensure_collection_exists"):
            result = vs.ingest_documents([], mock_weaviate_client)

        assert result == 0


class TestSimilaritySearch:
    """Tests for similarity_search()."""

    def test_returns_documents_from_store(self, mock_weaviate_client):
        """similarity_search should return whatever the store returns."""
        expected = [Document(page_content="Location info", metadata={})]
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = expected

        with patch("src.database.vector_store.get_vector_store", return_value=mock_store):
            result = vs.similarity_search("where are you located?", mock_weaviate_client, k=3)

        assert result == expected
        mock_store.similarity_search.assert_called_once_with("where are you located?", k=3)

    def test_no_results_returns_empty_list(self, mock_weaviate_client):
        """If the store returns nothing, an empty list should be returned."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        with patch("src.database.vector_store.get_vector_store", return_value=mock_store):
            result = vs.similarity_search("random query", mock_weaviate_client)

        assert result == []

    def test_k_parameter_is_passed_through(self, mock_weaviate_client):
        """The k parameter must be forwarded to the store."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        with patch("src.database.vector_store.get_vector_store", return_value=mock_store):
            vs.similarity_search("query", mock_weaviate_client, k=10)

        _, call_kwargs = mock_store.similarity_search.call_args
        assert call_kwargs.get("k") == 10 or mock_store.similarity_search.call_args[0][1:] == (10,) \
            or "k=10" in str(mock_store.similarity_search.call_args)


class TestDeleteAllDocuments:
    """Tests for delete_all_documents()."""

    def test_deletes_and_recreates_collection(self, mock_weaviate_client):
        """delete_all_documents should delete the collection then recreate it."""
        mock_weaviate_client.collections.exists.return_value = True

        with patch("src.database.vector_store.ensure_collection_exists") as mock_ensure:
            vs.delete_all_documents(mock_weaviate_client)

        mock_weaviate_client.collections.delete.assert_called_once_with(vs.COLLECTION_NAME)
        mock_ensure.assert_called_once()

    def test_no_delete_if_collection_missing(self, mock_weaviate_client):
        """If the collection doesn't exist, delete should not be called."""
        mock_weaviate_client.collections.exists.return_value = False

        with patch("src.database.vector_store.ensure_collection_exists"):
            vs.delete_all_documents(mock_weaviate_client)

        mock_weaviate_client.collections.delete.assert_not_called()
