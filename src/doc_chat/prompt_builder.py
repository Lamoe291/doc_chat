from doc_chat.models import SearchResult


PROMPT_TEMPLATE = """\
You are a helpful assistant answering questions about a provided document.

Use only the information in the context below to answer the question.
If the answer cannot be found in the context, say that you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

class PromptBuilder:
    def build(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str:
        context = "\n\n".join(
            f"[Source: {result.chunk.source}, "
            f"Page: {result.chunk.page_number}]\n"
            f"{result.chunk.text}"
            for result in results
        )

        return PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )