import numpy as np
import faiss
import tempfile
from pathlib import Path
from doc_chat.index_store import IndexStore
from doc_chat.vector_store import VectorStore
from doc_chat.models import Chunk


import pytest

def test_directory_creation_on_save(tmp_path: Path):
    embedding_dimension = 4
    chunks = [Chunk(id=str(i), source="fake_source", page_number=i, text=f"text {i}") for i in range(3)]
    rng = np.random.default_rng(42)

    embeddings = rng.random(
        (3, embedding_dimension),
        dtype=np.float32,
    )
    #embeddings = np.random.rand(3, embedding_dimension).astype(np.float32)
    vector_store = VectorStore(embedding_dimension=embedding_dimension)
    vector_store.add(chunks, embeddings)

    index_store = IndexStore()
    #with tempfile.TemporaryDirectory() as tmpdir:
    dir_path = tmp_path / "directory" #/ "index.faiss"
    index_store.save(vector_store, dir_path)
    assert dir_path.exists()
    assert (dir_path / "index.faiss").exists()
    assert (dir_path / "chunks.json").exists()

def test_save_and_load_index_store(tmp_path: Path): 
    embedding_dimension = 4
    chunks = [Chunk(id=str(i), source=Path("fake_source"), page_number=i, text=f"text {i}") for i in range(3)]
    rng = np.random.default_rng(42)
    embeddings = rng.random(
        (3, embedding_dimension),
        dtype=np.float32,
    )
    #embeddings = np.random.rand(3, embedding_dimension).astype(np.float32)
    vector_store = VectorStore(embedding_dimension=embedding_dimension)
    vector_store.add(chunks, embeddings)

    index_store = IndexStore()
    #with tempfile.TemporaryDirectory() as tmpdir:
    index_path = tmp_path / "index.faiss"
    index_store.save(vector_store, index_path)
    loaded_vector_store = index_store.load(index_path)

    assert loaded_vector_store.embedding_dimension == vector_store.embedding_dimension
    assert len(loaded_vector_store.chunks) == len(vector_store.chunks)
    for original_chunk, loaded_chunk in zip(vector_store.chunks, loaded_vector_store.chunks):
        assert original_chunk.id == loaded_chunk.id
        assert original_chunk.text == loaded_chunk.text
        assert original_chunk.source == loaded_chunk.source
    assert loaded_vector_store.index.ntotal == vector_store.index.ntotal
    

def test_retrieval_after_save_and_load(tmp_path: Path):
    embedding_dimension = 4
    chunks = [Chunk(id=str(i), source="fake_source", page_number=i, text=f"text {i}") for i in range(3)]
    rng = np.random.default_rng(42)
    embeddings = rng.random(
        (3, embedding_dimension),
        dtype=np.float32,
    )
    #embeddings = np.random.rand(3, embedding_dimension).astype(np.float32)
    vector_store = VectorStore(embedding_dimension=embedding_dimension)
    vector_store.add(chunks, embeddings)

    query_embedding = np.random.rand(1, embedding_dimension).astype(np.float32)

    results_before_save = vector_store.search(query_embedding=query_embedding, top_k=2)

    index_store = IndexStore()
    #with tempfile.TemporaryDirectory() as tmpdir:
    index_path = tmp_path / "index.faiss"
    index_store.save(vector_store, index_path)
    loaded_vector_store = index_store.load(index_path)

    results_after_load = loaded_vector_store.search(query_embedding=query_embedding, top_k=2)

    assert [r.chunk.id for r in results_after_load] == [r.chunk.id for r in results_before_save]
    assert [r.score for r in results_after_load] == [r.score for r in results_before_save]