from pathlib import Path
from doc_chat.rag_pipeline import RAGPipeline
from doc_chat.models import SearchResult, Chunk


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        chunk = Chunk(
            id="chunk_1",
            source=Path("test_doc.pdf"),
            page_number=1,
            text="Attention is a mechanism that allows models to focus on relevant parts.",
        )
        return [SearchResult(chunk=chunk, score=0.95)]


class FakePromptBuilder:
    def build(self, query: str, results: list[SearchResult]) -> str:
        return f"Query: {query}\nResults: {len(results)}"


class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        return "test answer"


def test_ask_runs_rag_flow():
    retriever = FakeRetriever()
    prompt_builder = FakePromptBuilder()
    llm = FakeLLMClient()

    pipeline = RAGPipeline(
        retriever,
        prompt_builder,
        llm,
    )

    answer = pipeline.ask("What is attention?", top_k=3)

    assert answer == "test answer"