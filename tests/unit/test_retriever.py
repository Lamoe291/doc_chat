from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from doc_chat.models import Chunk, SearchResult
from doc_chat.retriever import Retriever


def make_result(chunk_id: str, score: float) -> SearchResult:
    chunk = Chunk(
        id=chunk_id,
        source=Path("sample.pdf"),
        page_number=1,
        text=f"chunk {chunk_id}",
    )
    return SearchResult(chunk=chunk, score=score)


def test_empty_queries_are_rejected() -> None:
    embedder = Mock()
    vector_store = Mock()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    for query in ["", "   ", "\n\t"]:
        with pytest.raises(ValueError, match="Query must not be empty"):
            retriever.retrieve(query)


def test_query_is_passed_to_embedder_correctly() -> None:
    query = "what is retrieval?"
    query_embedding = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.search.return_value = []

    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    retriever.retrieve(query, top_k=2)

    embedder.embed.assert_called_once_with([query])


def test_resulting_embedding_is_passed_to_vector_store() -> None:
    query = "hello"
    query_embedding = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)

    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.search.return_value = []

    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    retriever.retrieve(query, top_k=4)

    vector_store.search.assert_called_once_with(query_embedding, top_k=4)


def test_top_k_is_forwarded_correctly() -> None:
    query_embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.search.return_value = []

    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    retriever.retrieve("query", top_k=11)

    _, kwargs = vector_store.search.call_args
    assert kwargs["top_k"] == 11


def test_vector_store_results_are_returned_unchanged() -> None:
    expected_results = [
        make_result("c1", 0.95),
        make_result("c2", 0.80),
    ]

    query_embedding = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.search.return_value = expected_results

    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    actual_results = retriever.retrieve("find relevant context", top_k=2)

    assert actual_results is expected_results
