from pathlib import Path

import pytest

from doc_chat.chunker import TextChunker
from doc_chat.models import Document


@pytest.fixture
def test_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "test.pdf"

    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>endobj\n"
        b"trailer<< /Root 1 0 R >>\n"
        b"%%EOF\n"
    )

    return pdf_path


@pytest.fixture
def test_document(test_pdf: Path) -> Document:
    return Document(source=test_pdf, page_number=1, text="Hello page one.")


def test_character_chunking_with_overlap():
    chunker = TextChunker()

    chunks = chunker.chunk(
        Document(source=Path("sample.pdf"), page_number=1, text="abcdefghij"),
        granularity="characters",
        chunk_size=4,
        chunk_overlap=1,
    )

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij"]


def test_word_chunking_with_overlap():
    chunker = TextChunker()

    chunks = chunker.chunk(
        Document(source=Path("sample.pdf"), page_number=1, text="one two three four five six"),
        granularity="words",
        chunk_size=3,
        chunk_overlap=1,
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_paragraph_chunking_with_overlap():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunker = TextChunker()

    chunks = chunker.chunk(
        Document(source=Path("sample.pdf"), page_number=1, text=text),
        granularity="paragraphs",
        chunk_size=2,
        chunk_overlap=1,
    )

    assert [chunk.text for chunk in chunks] == [
        "First paragraph.\n\nSecond paragraph.",
        "Second paragraph.\n\nThird paragraph.",
    ]


def test_chunk_returns_empty_for_empty_text():
    chunker = TextChunker()

    chunks = chunker.chunk(
        Document(source=Path("sample.pdf"), page_number=1, text=""),
        granularity="words",
        chunk_size=2,
        chunk_overlap=0,
    )

    assert chunks == []


def test_invalid_granularity_raises_value_error():
    chunker = TextChunker()

    with pytest.raises(ValueError):
        chunker.chunk(
            Document(source=Path("sample.pdf"), page_number=1, text="text"),
            granularity="sentences",
            chunk_size=2,
            chunk_overlap=0,
        )


def test_chunker_can_use_document_from_dummy_pdf(test_document: Document):
    chunker = TextChunker()

    chunks = chunker.chunk(
        test_document,
        granularity="words",
        chunk_size=2,
        chunk_overlap=0,
    )

    assert [chunk.text for chunk in chunks] == ["Hello", "page", "one."]
