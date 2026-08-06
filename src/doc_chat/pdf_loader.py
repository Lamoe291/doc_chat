from pathlib import Path
from doc_chat.models import Document
import re
from doc_chat.text_cleaner import clean_text
import pymupdf



class PDFLoader:
    """ Loader for PDF documents. """

    def load(self, pdf_path: Path) -> list[Document]:
        """ Load a PDF document and split it into pages. """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {pdf_path}")

        with pymupdf.open(pdf_path) as pdf:
            documents = []
            for i, page in enumerate(pdf, start=0):
                text = page.get_text()
                text = clean_text(text)
                if text:
                    documents.append(Document(source=pdf_path, page_number=i+1, text=text))
        return documents
    

