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

    def index(self, pdf_path: str) -> None:
        # Load the PDF and extract text
        text = self.loader.load(pdf_path)
        # Split the text into chunks
        chunks = self.chunker.chunk(text)
        # Embed the chunks
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        # Add the chunks and their embeddings to the vector store
        self.vector_store.add(chunks, embeddings)