import pytest
from app.core.prompts import(
    summarize_for_engineer,
    rag_answer,
    few_shot_classifier,
    chain_of_thought,
    extract_json,
)

def test_summarize_includes_article():
    article = "FastAPI is a pythom web framework."
    prompt = summarize_for_engineer(article)
    assert article in prompt
    assert isinstance(prompt, str)

def test_summarize_has_instructions():
    prompt = summarize_for_engineer("Some article text")
    assert "tradeoffs" in prompt.lower()



#rag test
def test_rag_includes_questions():
    prompt = rag_answer("What is FastAPI?", ["FastAPI is a frameowork"])
    assert "What is FastAPI?" in prompt

def test_rag_includes_all_chunks():
    chunks = ["Chunk one content", "Chunk two content"]
    prompt = rag_answer("question", chunks)
    assert "Chunk one content" in prompt
    assert "Chunk two content" in prompt

def test_rag_has_hallucination_gaurd():
    prompt = rag_answer("question", ["context"])

    assert "I don't know based on the provided context" in prompt

def test_rag_empty_chunks():
    prompt = rag_answer("question", [])
    assert isinstance(prompt, str)

def test_few_shot_includes_examples():
    examples = [
        {"text": "Great product!", "label":"positive"},
        {"text": "Terrible quality", "label": "negative"},

    ]
    prompt = few_shot_classifier("Amazing service", examples)
    assert "Great product!" in prompt
    assert "positive" in prompt
    assert "Amazing service" in prompt

def test_chain_of_thought_includes_problem():
    prompt = chain_of_thought("What is 15% of 200?")
    assert "What is 15% of 200?" in prompt
    assert "step by step" in prompt.lower()


def text_extract_json_includes_text_and_schema():
    prompt = extract_json(
        text="John Smith, age 30, works at Anthropic",
        schema_description = '{"name":"string", "age":"number", "company":"string"}'
        
    )
    assert "John Smith" in prompt
    assert "name" in prompt
    assert "JSON" in prompt