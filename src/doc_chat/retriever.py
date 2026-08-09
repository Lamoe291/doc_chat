
from doc_chat.embedder import Embedder
from doc_chat.vector_store import VectorStore
from doc_chat.models import SearchResult

class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        embedded_query = self.embedder.embed([query])
        return self.vector_store.search(embedded_query, top_k=top_k)