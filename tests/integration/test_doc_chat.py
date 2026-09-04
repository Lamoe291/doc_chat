from pathlib import Path

import numpy as np
from reportlab.pdfgen import canvas

from doc_chat.chunker import TextChunker
from doc_chat.embedder import Embedder
from doc_chat.index_store import IndexStore
from doc_chat.models import Chunk
from doc_chat.pdf_loader import PDFLoader
from doc_chat.prompt_builder import PromptBuilder
from doc_chat.rag_pipeline import RAGPipeline
from doc_chat.retriever import Retriever
from doc_chat.vector_store import VectorStore


def create_test_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 750, "Test Document")
    pdf.drawString(72, 730, "The capital of France is Paris.")
    pdf.drawString(72, 710, "The capital of Germany is Berlin.")
    pdf.save()


class FakeEmbedder:
    def embed(self, texts):
        vectors = []

        for text in texts:

            if "france" in text or "paris" in text:
                vectors.append([1.0, 0.0])
            elif "germany" in text or "berlin" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.0])

        return np.asarray(vectors, dtype=np.float32)


class FakeLLMClient:
    def generate(self, prompt):
        return "The capital of France is Paris."


def test_rag_pipeline_end_to_end(tmp_path):
    pdf_path = tmp_path / "test_document.pdf"
    index_directory = tmp_path / "index"

    create_test_pdf(pdf_path)

    loader = PDFLoader()
    chunker = TextChunker(granularity="words", chunk_size=3, chunk_overlap=1)

    pages = loader.load(pdf_path)

    chunks = []
    for page in pages:
        chunks.extend(
            chunker.chunk(
                page
            )
        )

    assert chunks
    assert any("Paris" in chunk.text for chunk in chunks)

    embedder = FakeEmbedder()
    embeddings = embedder.embed([chunk.text for chunk in chunks])

    vector_store = VectorStore(
        embedding_dimension=2,
    )
    vector_store.add(chunks, embeddings)

    index_store = IndexStore()
    index_store.save(vector_store, index_directory)

    loaded_store = index_store.load(index_directory)

    assert loaded_store.index.ntotal == len(chunks)
    assert len(loaded_store.chunks) == len(chunks)

    retriever = Retriever(
        embedder=embedder,
        vector_store=loaded_store,
    )

    prompt_builder = PromptBuilder()
    llm_client = FakeLLMClient()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    answer = pipeline.ask("What is the capital of France?")

    assert answer == "The capital of France is Paris."