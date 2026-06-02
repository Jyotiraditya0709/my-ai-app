import pytest
from evals.rag_eval import eval_retrieval, GOLDEN

@pytest.mark.asyncio
async def test_retrieval_hit_rate_above_threshold():
    """Regression gaurd - fails if retrieval quality drops below 66%

    regression gaurd = a test that prevents you from accidentally 
    making the system worse when you change chunking, embeddings, or prompts.
    If this test fails after a change, your change broke retrieval.
    """

    with open("data/sample_doc.md") as f:
        await rag.ingest(f.read(), source= "sample_doc.md") 
    
    results = await eval_retrieval(GOLDEN)

    assert result["hit_rate"] >= 0.66, f"Retrieval too low: {result}"

    await rag.store.delete_by_source("sample_doc.md")
    await rag.store.close()
    