import pytest
import numpy as np
from pathlib import Path
from doc_chat.embedder import Embedder
from doc_chat.models import Chunk

TEST_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def make_chunk(text: str) -> Chunk:
        return Chunk(
            id="test_chunk",
            source=Path("test.pdf"),
            page_number=1,
            text=text,
        )

@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(TEST_MODEL_NAME)

def test_embedder_returns_expected_shape(embedder):

    chunks = [make_chunk("Hello world"), make_chunk("Goodbye world")]

    embeddings = embedder.embed(chunks)

    assert embeddings.shape == (len(chunks), 384)

def test_embedder_returns_float32(embedder):
    chunks = [
        make_chunk("Hello world"),
        make_chunk("Goodbye world"),
    ]

    embeddings = embedder.embed(chunks)

    assert embeddings.dtype == np.float32

def test_embedder_returns_normalized_embeddings(embedder):
    
    chunks = [make_chunk("Hello world"), make_chunk("Goodbye world")]
    
    embeddings = embedder.embed(chunks)
    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0)

def test_semantically_similar_text_has_higher_similarity(embedder):
    chunks = [
        make_chunk("A cat is sleeping on the sofa."),
        make_chunk("A kitten is resting on the couch."),
        make_chunk("The stock market dropped significantly today.")
    ]
    embeddings = embedder.embed(chunks)
    similarity_related = np.dot(embeddings[0], embeddings[1])
    similarity_unrelated = np.dot(embeddings[0], embeddings[2])

    assert similarity_related > similarity_unrelated

