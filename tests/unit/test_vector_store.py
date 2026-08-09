from pathlib import Path

import numpy as np
import pytest

from doc_chat.models import Chunk, SearchResult
from doc_chat.vector_store import VectorStore


def make_chunk(chunk_id: str, text: str) -> Chunk:
	return Chunk(
		id=chunk_id,
		source=Path("sample.pdf"),
		page_number=1,
		text=text,
	)


def test_add_three_chunks_creates_three_vectors_in_index() -> None:
	store = VectorStore(embedding_dimension=3)
	chunks = [
		make_chunk("c1", "chunk 1"),
		make_chunk("c2", "chunk 2"),
		make_chunk("c3", "chunk 3"),
	]
	embeddings = np.asarray(
		[
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)

	store.add(chunks, embeddings)

	assert store.index.ntotal == 3


def test_search_returns_expected_count_and_ordered_by_similarity() -> None:
	store = VectorStore(embedding_dimension=3)
	chunks = [
		make_chunk("c1", "most similar"),
		make_chunk("c2", "second most similar"),
		make_chunk("c3", "least similar"),
	]
	embeddings = np.asarray(
		[
			[1.0, 0.0, 0.0],
			[0.8, 0.2, 0.0],
			[0.0, 1.0, 0.0],
		],
		dtype=np.float32,
	)
	store.add(chunks, embeddings)
	query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

	results = store.search(query, top_k=3)

	assert len(results) == 3
	assert [result.chunk.id for result in results] == ["c1", "c2", "c3"]
	assert results[0].score >= results[1].score >= results[2].score


def test_search_returns_searchresult_with_matching_chunk() -> None:
	store = VectorStore(embedding_dimension=3)
	chunk = make_chunk("unique", "target chunk")
	store.add([chunk], np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
	query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

	results = store.search(query, top_k=1)

	assert len(results) == 1
	assert isinstance(results[0], SearchResult)
	assert results[0].chunk == chunk


def test_top_k_larger_than_store_size_returns_all_available_results_once() -> None:
	store = VectorStore(embedding_dimension=3)
	chunks = [
		make_chunk("c1", "alpha"),
		make_chunk("c2", "beta"),
		make_chunk("c3", "gamma"),
	]
	embeddings = np.asarray(
		[
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)
	store.add(chunks, embeddings)
	query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

	results = store.search(query, top_k=10)

	assert len(results) == 3
	assert len({result.chunk.id for result in results}) == 3


def test_empty_store_search_returns_empty_list() -> None:
	store = VectorStore(embedding_dimension=3)
	query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

	results = store.search(query, top_k=5)

	assert results == []

def test_search_maps_index_to_correct_chunk():
    store = VectorStore(embedding_dimension=3)

    chunks = [
        make_chunk("first", "first chunk"),
        make_chunk("second", "second chunk"),
        make_chunk("third", "third chunk"),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.add(chunks, embeddings)

    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    results = store.search(query, top_k=1)

    assert results[0].chunk is chunks[1]
    assert results[0].score == pytest.approx(1.0)
