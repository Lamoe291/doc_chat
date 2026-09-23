from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent

@dataclass
class Settings:
    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    document_dir = data_dir / "documents"
    vectorstore_dir = PROJECT_ROOT / "vectorstores"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    chunk_size: int = 200
    chunk_overlap: int = 50
    top_k: int = 10

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")


settings = Settings()