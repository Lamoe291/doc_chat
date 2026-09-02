
import pytest
from doc_chat.llm_client import LLMClient

TEST_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

@pytest.fixture(scope="module")
def llm() -> LLMClient:
    return LLMClient(TEST_MODEL_NAME)

def test_llm_generates_text(llm):
    #llm = LLMClient(TEST_MODEL_NAME)

    result = llm.generate(
        "What is 2 + 2? Answer briefly."
    )

    assert isinstance(result, str)
    assert result.strip()