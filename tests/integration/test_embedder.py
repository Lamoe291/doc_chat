import pytest
import numpy as np
from pathlib import Path
from doc_chat.embedder import Embedder
from doc_chat.models import Chunk

TEST_MODEL_NAME = "BAAI/bge-small-en-v1.5"

@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(TEST_MODEL_NAME)

def test_embedder_returns_expected_shape(embedder):

    texts = ["Hello world", "Goodbye world"]
    embeddings = embedder.embed(texts)

    assert embeddings.shape == (len(texts), 384)

def test_embedder_returns_float32(embedder):
    texts = ["Hello world", "Goodbye world"]

    embeddings = embedder.embed(texts)

    assert embeddings.dtype == np.float32

def test_embedder_returns_normalized_embeddings(embedder):
    
    texts = ["Hello world", "Goodbye world"]

    embeddings = embedder.embed(texts)
    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0)

def test_semantically_similar_text_has_higher_similarity(embedder):
    texts = [
        "A cat is sleeping on the sofa.",
        "A kitten is resting on the couch.",
        "The stock market dropped significantly today."
    ]
    embeddings = embedder.embed(texts)
    similarity_related = np.dot(embeddings[0], embeddings[1])
    similarity_unrelated = np.dot(embeddings[0], embeddings[2])

    assert similarity_related > similarity_unrelated

