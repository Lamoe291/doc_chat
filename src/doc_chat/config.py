from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATADIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATADIR / "documents"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstores"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"