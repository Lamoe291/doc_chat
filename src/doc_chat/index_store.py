from pathlib import Path
from dataclasses import asdict
from doc_chat.models import Chunk
import json
import faiss
from doc_chat.vector_store import VectorStore

class IndexStore:

    def save(self, vector_store: VectorStore, path: Path) -> None:
        

        path.mkdir(parents=True, exist_ok=True)

        index_path = path / "index.faiss"
        chunks_path = path / "chunks.json"

        faiss.write_index(vector_store.index, str(index_path))

        # Save the chunks associated with the vector store
        with open(chunks_path, "w") as f:
            json.dump([{**asdict(chunk),
        "source": str(chunk.source)} for chunk in vector_store.chunks], f)


    def load(self, path: Path) -> VectorStore:
        index_path = path / "index.faiss"
        chunks_path = path / "chunks.json"

        index = faiss.read_index(str(index_path))
        with open(chunks_path, "r") as f:
            loaded_chunks = json.load(f)
            chunks = [Chunk(id=chunk_dict["id"],
        source=Path(chunk_dict["source"]),
        page_number=chunk_dict["page_number"],
        text=chunk_dict["text"]) for chunk_dict in loaded_chunks]
        return VectorStore(embedding_dimension=index.d, index=index, chunks=chunks)