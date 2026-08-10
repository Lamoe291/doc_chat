from pathlib import Path

from doc_chat.models import Chunk, SearchResult
from doc_chat.prompt_builder import PromptBuilder


def make_result(text: str, source: str = "sample.pdf", page_number: int = 1) -> SearchResult:
    chunk = Chunk(
        id="chunk-1",
        source=Path(source),
        page_number=page_number,
        text=text,
    )
    return SearchResult(chunk=chunk, score=0.99)


def test_prompt_contains_query():
    builder = PromptBuilder()
    prompt = builder.build("What is attention?", [make_result("attention mechanism")])

    assert "What is attention?" in prompt


def test_prompt_contains_chunk_text_and_metadata():
    builder = PromptBuilder()
    prompt = builder.build("What is attention?", [make_result("attention mechanism", "doc.pdf", 3)])

    assert "attention mechanism" in prompt
    assert "[Source: doc.pdf, Page: 3]" in prompt


def test_prompt_includes_multiple_results_in_context():
    builder = PromptBuilder()
    results = [
        make_result("attention mechanism", "doc1.pdf", 1),
        make_result("transformers", "doc2.pdf", 2),
    ]

    prompt = builder.build("What is attention?", results)

    assert "[Source: doc1.pdf, Page: 1]" in prompt
    assert "attention mechanism" in prompt
    assert "[Source: doc2.pdf, Page: 2]" in prompt
    assert "transformers" in prompt
    assert prompt.index("attention mechanism") < prompt.index("transformers")