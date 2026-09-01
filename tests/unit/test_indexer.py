from pathlib import Path
import pytest
from doc_chat.models import Chunk
from doc_chat.models import Document

from doc_chat.indexer import Indexer

class FakePDFLoader:
    def load(self, pdf_path: Path):
        return [
            Document(source=pdf_path, page_number=1, text="Page 1 text"),
            Document(source=pdf_path, page_number=2, text="Page 2 text"),
        ]

class FakeChunker:
    def chunk(self, text: str):
        return [Chunk(id=f"{text}-chunk-{i}", source="fake_source", page_number=i, text=f"{text} chunk {i}") for i in range(2)]

class FakeEmbedder:
    def embed(self, texts: list[str]):
        return [[0.1, 0.2, 0.3] for _ in texts]

class FakeVectorStore:
    def __init__(self):
        self.store = []

    def add(self, chunks, embeddings):
        for chunk, embedding in zip(chunks, embeddings):
            self.store.append((chunk, embedding))


def test_indexer():
    indexer = Indexer(
        loader=FakePDFLoader(),
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    indexer.index(Path("sample.pdf"))
    #print(indexer.vector_store.store)
    assert len(indexer.vector_store.store) == 4  # 2 pages * 2 chunks per page

