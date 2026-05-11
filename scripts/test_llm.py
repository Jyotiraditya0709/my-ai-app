import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.llm import llm
from app.core.prompts import summarize_for_engineer, rag_answer
from pydantic import BaseModel

class Sentiment(BaseModel):
    sentiment: str
    confidence: float

async def main():
    article = "FastAPI is a modern Python web framework. Key tradeoffs: it's async-native, validates requests automatically, but has a smaller ecosystem than Django."

    # 1. Basic completion
    print("=== 1. Basic completion (gpt-4o-mini) ===")
    res = await llm.complete(summarize_for_engineer(article))
    print(f"Content: {res.content}")
    print(f"Tokens: {res.input_tokens} in + {res.output_tokens} out")
    print(f"Cost: ${res.cost_usd:.6f} | Latency: {res.latency_ms}ms\n")

    # # 2. Same prompt, Claude
    # print("=== 2. Same prompt with Claude ===")
    # res2 = await llm.complete(
    #     summarize_for_engineer(article),
    #     model="claude-3-haiku-20240307",
    # )
    # print(f"Content: {res2.content}")
    # print(f"Cost: ${res2.cost_usd:.6f} | Latency: {res2.latency_ms}ms\n")

    # 3. Streaming
    print("=== 3. Streaming ===")
    async for token in llm.stream("Count from 1 to 5 slowly, one per line."):
        print(token, end="", flush=True)
    print("\n")

    # 4. Structured output
    print("=== 4. Structured output ===")
    result = await llm.complete_structured(
        "Classify sentiment: This product saved my weekend!",
        schema=Sentiment,
    )
    print(f"Type: {type(result).__name__}")
    print(f"Sentiment: {result.sentiment}, Confidence: {result.confidence}\n")

    # 5. RAG-style prompt
    print("=== 5. RAG-style with hallucination guard ===")
    chunks = [
        "FastAPI endpoints are defined with decorators like @app.get().",
        "Pydantic v2 validates request bodies automatically.",
    ]
    res3 = await llm.complete(rag_answer(
        "How do I add authentication to FastAPI?",
        context_chunks=chunks,
    ))
    print(f"Answer: {res3.content}")
    print("(Should say it doesn't know — the chunks don't cover auth)")

asyncio.run(main())