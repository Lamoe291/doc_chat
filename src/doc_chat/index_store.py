from pathlib import Path
from dataclasses import asdict
from doc_chat.models import Chunk
import json
import faiss
from doc_chat.vector_store import VectorStore

class IndexStore:

    def save(self, vector_store: VectorStore, path: Path) -> None:
        faiss.write_index(vector_store.index, str(path))
        # Save the chunks associated with the vector store
        with open(path.with_suffix(".chunks"), "w") as f:
            json.dump([asdict(chunk) for chunk in vector_store.chunks], f)


    def load(self, path: Path) -> VectorStore:
        index = faiss.read_index(str(path))
        with open(path.with_suffix(".chunks"), "r") as f:
            loaded_chunks = json.load(f)
            chunks = [Chunk(**chunk_dict) for chunk_dict in loaded_chunks]
        return VectorStore(embedding_dimension=index.d, index=index, chunks=chunks)