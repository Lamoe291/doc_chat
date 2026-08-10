
from doc_chat.llm_client import LLMClient

TEST_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
def test_llm_generates_text():
    llm = LLMClient(TEST_MODEL_NAME)

    result = llm.generate(
        "What is 2 + 2? Answer briefly."
    )

    assert isinstance(result, str)
    assert result.strip()