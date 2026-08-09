import re
from typing import Literal

from doc_chat.models import Chunk, Document

class TextChunker:
    """Chunk documents with configurable granularity and overlap."""

    def __init__(
        self,
        granularity: Literal["characters", "words", "paragraphs"] = "characters",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        if granularity not in {"characters", "words", "paragraphs"}:
            raise ValueError(
                "granularity must be one of: characters, words, paragraphs"
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.granularity = granularity
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        """Chunk a document into smaller pieces."""

        if not doc.text:
            return []

        if self.granularity == "characters":
            chunk_texts = self._chunk_characters(doc.text)
        elif self.granularity == "words":
            words = doc.text.split()
            chunk_texts = self._chunk_units(words, joiner=" ")
        else:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", doc.text) if p.strip()]
            chunk_texts = self._chunk_units(paragraphs, joiner="\n\n")

        return [
            Chunk.from_document(doc, i, text)
            for i, text in enumerate(chunk_texts)
        ]

    def _chunk_characters(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += step

        return chunks

    def _chunk_units(self, units: list[str], joiner: str) -> list[str]:
        if not units:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0

        while start < len(units):
            end = min(start + self.chunk_size, len(units))
            chunks.append(joiner.join(units[start:end]))
            if end == len(units):
                break
            start += step

        return chunks


 