from doc_chat.retriever import Retriever
from doc_chat.prompt_builder import PromptBuilder
from doc_chat.llm_client import LLMClient

class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def ask(self, query: str, top_k: int = 5) -> str:
        results = self.retriever.retrieve(query, top_k=top_k)
        prompt = self.prompt_builder.build(query, results)
        return self.llm_client.generate(prompt)