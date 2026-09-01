from pathlib import Path
from doc_chat.pdf_loader import PDFLoader
from doc_chat.chunker import TextChunker
from doc_chat.embedder import Embedder
from doc_chat.vector_store import VectorStore


class Indexer:

    def __init__(self, loader: PDFLoader, chunker: TextChunker, embedder: Embedder, vector_store: VectorStore) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index(self, pdf_path: Path) -> None:
        # Load the PDF and extract text
        pages = self.loader.load(pdf_path)
        # Split the text into chunks
        for page in pages:
            chunks = self.chunker.chunk(page)
            if not chunks:
                continue
            # Embed the chunks
            embeddings = self.embedder.embed([chunk.text for chunk in chunks])
            # Add the chunks and their embeddings to the vector store
            self.vector_store.add(chunks, embeddings)