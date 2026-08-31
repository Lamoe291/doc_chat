from doc_chat.pdf_loader import PDFLoader
from doc_chat.chunker import TextChunker
from doc_chat.embedder import Embedder
from doc_chat.vector_store import VectorStore
from doc_chat.retriever import Retriever
from doc_chat.llm_client import LLMClient
from doc_chat.prompt_builder import PromptBuilder
from doc_chat.rag_pipeline import RAGPipeline
from doc_chat.indexer import Indexer
from doc_chat.config import EMBEDDING_BACKBONE_NAME, LLM_MODEL_NAME

def run(pdf_path: str) -> None:
    # Initialize components
    loader = PDFLoader(pdf_path)
    chunker = TextChunker(granularity="words", chunk_size=200, chunk_overlap=50)  # Adjust granularity as needed
    embedder = Embedder(backbone_name=EMBEDDING_BACKBONE_NAME)
    vector_store = VectorStore(embedding_dim=embedder.backbone.get_sentence_embedding_dimension())
    retriever = Retriever(embedder, vector_store)
    prompt_builder = PromptBuilder()
    llm_client = LLMClient(model_name=LLM_MODEL_NAME)

    # Index the PDF
    indexer = Indexer(loader, chunker, embedder, vector_store)
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(retriever, prompt_builder, llm_client)

    indexer.index(pdf_path)

    print(f"Indexed {vector_store.index.ntotal} chunks.")
    print("Ask a question (or type 'exit' to quit):")

    

    while True:
        query = input("> ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        answer = pipeline.ask(query)
        print(f"\n{answer}\n")