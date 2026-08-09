from pathlib import Path
import pytest
import pymupdf

from doc_chat.pdf_loader import PDFLoader


@pytest.fixture
def test_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "test.pdf"

    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Hello page one."
    )

    doc.save(pdf_path)
    doc.close()

    return pdf_path


def test_load_pdf(test_pdf):
    loader = PDFLoader()

    documents = loader.load(test_pdf)

    assert len(documents) == 1
    assert documents[0].text == "Hello page one."