import numpy as np
import faiss
from doc_chat.embedder import Embedder
from doc_chat.models import Chunk, SearchResult




class VectorStore:
    def __init__(self, embedding_dimension: int, index: faiss.IndexFlatIP | None = None, chunks: list[Chunk] | None = None):
        self.embedding_dimension = embedding_dimension
        self.index = index if index is not None else faiss.IndexFlatIP(self.embedding_dimension)
        #self.vectors: np.ndarray = np.empty((0, self.embedding_dimension), dtype=np.float32)
        self.chunks = chunks if chunks is not None else []
        

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dimension}, "
                f"got {embeddings.shape[1]}"
            )
        self.chunks.extend(chunks)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if self.index.ntotal == 0:
            return []

        if top_k <= 0:
            return []

        if query_embedding.shape[-1] != self.embedding_dimension:
            raise ValueError(
                f"Expected query embedding dimension "
                f"{self.embedding_dimension}, "
                f"got {query_embedding.shape[-1]}"
            )
        query_embedding = query_embedding.reshape(1, -1)
        effective_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, effective_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results
