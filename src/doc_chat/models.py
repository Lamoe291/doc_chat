from dataclasses import dataclass
from pathlib import Path

@dataclass(slots=True)
class Document:
    source: Path
    page_number: int
    text: str

@dataclass(slots=True)
class Chunk:
    id: str
    source: Path
    page_number: int
    text: str

    @property
    def location(self) -> str:
        return f"{self.source.name}, page {self.page_number}"


    @staticmethod
    def _create_chunk_id(
        doc: Document,
        chunk_index: int,
    ) -> str:
        return (
            f"{doc.source.stem}"
            f"_p{doc.page_number}"
            f"_c{chunk_index}"
        )

    @classmethod
    def from_document(cls, doc: Document, chunk_index: int, chunk_text: str) -> "Chunk":
        return cls(
            id=cls._create_chunk_id(doc, chunk_index),
            source=doc.source,
            page_number=doc.page_number,
            text=chunk_text,
        )

@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float