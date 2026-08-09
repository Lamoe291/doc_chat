from pathlib import Path

import pytest

from doc_chat.chunker import TextChunker
from doc_chat.models import Document


def make_document(text: str) -> Document:
    return Document(source=Path("sample.pdf"), page_number=1, text=text)


def test_character_chunking_with_overlap():
    chunker = TextChunker(granularity="characters",
        chunk_size=4,
        chunk_overlap=1)

    chunks = chunker.chunk(
        make_document("abcdefghij")
    )

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij"]


def test_word_chunking_with_overlap():
    chunker = TextChunker(granularity="words",
        chunk_size=3,
        chunk_overlap=1)

    chunks = chunker.chunk(
        make_document("one two three four five six")
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_paragraph_chunking_with_overlap():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunker = TextChunker(granularity="paragraphs",
        chunk_size=2,
        chunk_overlap=1)

    chunks = chunker.chunk(
        make_document(text)
    )

    assert [chunk.text for chunk in chunks] == [
        "First paragraph.\n\nSecond paragraph.",
        "Second paragraph.\n\nThird paragraph.",
    ]


def test_chunk_returns_empty_for_empty_text():
    chunker = TextChunker(granularity="words",
        chunk_size=2,
        chunk_overlap=0)

    chunks = chunker.chunk(
        make_document("")
    )

    assert chunks == []


def test_invalid_granularity_raises_value_error():
    
    with pytest.raises(ValueError):
        chunker = TextChunker(granularity="InvalidGranularity")
        #chunker.chunk(make_document("text"))
