import numpy as np
import faiss
import tempfile
from pathlib import Path
from doc_chat.index_store import IndexStore
from doc_chat.vector_store import VectorStore
from doc_chat.models import Chunk


import pytest

def test_save_and_load_index_store():
    embedding_dimension = 4
    chunks = [Chunk(id=str(i), source="fake_source", page_number=i, text=f"text {i}") for i in range(3)]
    embeddings = np.random.rand(3, embedding_dimension).astype(np.float32)
    vector_store = VectorStore(embedding_dimension=embedding_dimension)
    vector_store.add(chunks, embeddings)

    index_store = IndexStore()
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index.faiss"
        index_store.save(vector_store, index_path)
        loaded_vector_store = index_store.load(index_path)

        assert loaded_vector_store.embedding_dimension == vector_store.embedding_dimension
        assert len(loaded_vector_store.chunks) == len(vector_store.chunks)
        for original_chunk, loaded_chunk in zip(vector_store.chunks, loaded_vector_store.chunks):
            assert original_chunk.id == loaded_chunk.id
            assert original_chunk.text == loaded_chunk.text
        assert loaded_vector_store.index.ntotal == vector_store.index.ntotal