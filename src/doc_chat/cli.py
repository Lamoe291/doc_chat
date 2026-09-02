from pathlib import Path
import argparse
from doc_chat.pdf_loader import PDFLoader
from doc_chat.chunker import TextChunker
from doc_chat.embedder import Embedder
from doc_chat.vector_store import VectorStore
from doc_chat.retriever import Retriever
from doc_chat.llm_client import LLMClient
from doc_chat.prompt_builder import PromptBuilder
from doc_chat.rag_pipeline import RAGPipeline
from doc_chat.indexer import Indexer
from doc_chat.index_store import IndexStore
from doc_chat.config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME

# def run(pdf_path: Path) -> None:
#     # Initialize components
#     loader = PDFLoader()
#     chunker = TextChunker(granularity="words", chunk_size=300, chunk_overlap=50)  # Adjust granularity as needed
#     embedder = Embedder(backbone_name=EMBEDDING_MODEL_NAME)
#     vector_store = VectorStore(embedding_dimension=embedder.backbone.get_embedding_dimension())
#     retriever = Retriever(embedder, vector_store)
#     prompt_builder = PromptBuilder()
#     llm_client = LLMClient(model_name=LLM_MODEL_NAME)
#     #print(llm_client.generator.generation_config)

#     # Index the PDF
#     indexer = Indexer(loader, chunker, embedder, vector_store)
#     # Create RAG pipeline
#     rag_pipeline = RAGPipeline(retriever, prompt_builder, llm_client)

#     indexer.index(pdf_path)

#     print(f"Indexed {vector_store.index.ntotal} chunks.")
#     print("Ask a question (or type 'exit' to quit):")

    

#     while True:
#         query = input("> ").strip()

#         if query.lower() == "exit":
#             break

#         if not query:
#             continue

#         answer = rag_pipeline.ask(query)
#         print(f"\n{answer}\n")

def index_command(args):
    loader = PDFLoader()
    chunker = TextChunker(granularity="words", chunk_size=300, chunk_overlap=50)  # Adjust granularity as needed
    embedder = Embedder(backbone_name=EMBEDDING_MODEL_NAME)
    vector_store = VectorStore(embedding_dimension=embedder.backbone.get_embedding_dimension())
    indexer = Indexer(loader, chunker, embedder, vector_store)

    indexer.index(args.pdf_path)
    print(f"Indexed {vector_store.index.ntotal} chunks.")

    index_store = IndexStore()
    index_name = args.pdf_path.stem
    index_directory = Path("data/indexes") / index_name
    index_store.save(vector_store, index_directory)

    #run(args.pdf_path)

def chat_command(args):
    embedder = Embedder(backbone_name=EMBEDDING_MODEL_NAME)
    index_store = IndexStore()
    index_directory = Path("data/indexes") / args.index
    vector_store = index_store.load(index_directory)
    print(f"Loaded {vector_store.index.ntotal} chunks from the index.")
    
    retriever = Retriever(embedder, vector_store)
    prompt_builder = PromptBuilder()
    llm_client = LLMClient(model_name=LLM_MODEL_NAME)
    rag_pipeline = RAGPipeline(retriever, prompt_builder, llm_client)

    print(f"Indexed {vector_store.index.ntotal} chunks.")
    print("Ask a question (or type 'exit' to quit):")
    
    while True:
        query = input("> ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        answer = rag_pipeline.ask(query)
        print(f"\n{answer}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Chat with your PDF documents."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -----------------
    # index
    # -----------------

    index_parser = subparsers.add_parser(
        "index",
        help="Index a PDF document.",
    )

    index_parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF document.",
    )

    index_parser.set_defaults(func=index_command)

    # -----------------
    # chat
    # -----------------

    chat_parser = subparsers.add_parser(
        "chat",
        help="Chat with the indexed document.",
    )

    chat_parser.add_argument(
        "--index",
        required=True,
        help="Name of the index to chat with.",
    )

    chat_parser.set_defaults(func=chat_command)

    # Parse arguments and execute command
    args = parser.parse_args(argv)
    args.func(args)