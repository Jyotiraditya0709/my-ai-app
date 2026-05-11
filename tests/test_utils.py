import pytest
from app.core.utils import truncate, count_tokens, build_prompt

@pytest.mark.parametrize("text, max_len, expected",[

    ("hello", 10, "hello"),
    ("hello world", 5, "hello..."),
    ("hi", 2, "hi"),
    ("", 5, ""),
])
def test_truncate_parametrized(text, max_len, expected):
    assert truncate(text, max_len)== expected


def test_count_tokens_returns_int():
    count = count_tokens("What is retrieval augmented generation?")
    assert isinstance(count, int)
    assert count > 0

def test_build_prompt_includes_question():
    prompt = build_prompt("Some context here", "What is RAG?")
    assert "What is RAG?" in prompt
    assert "Some context here" in prompt

def test_build_prompt_raises_when_too_long():
    with pytest.raises(ValueError, match="too long"):
        build_prompt("x " * 10000, "question")  # "x " repeated = more tokens