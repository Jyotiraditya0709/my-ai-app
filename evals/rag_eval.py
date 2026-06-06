from dataclasses import dataclass

import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.rag import rag
from pydantic import BaseModel
from typing import Literal

# ------EVAL CASE-----------------------
@dataclass
class RAGEvalCase:
    question: str
    expected_facts : list[str]
    expected_source: str
    should_refuse: bool = False


#------Faithfulness verdics-----------------
class FaithfulnessVerdict(BaseModel):
    verdict: Literal["supported", "partial","unsupported"]
    unsupported_claims: list[str]
    reasoning: str


#-------Golden DataSet---------------------
GOLDEN = [
    RAGEvalCase(
        "How does Pydantic validation work in FastAPI?",
        ["validation", "request"],
        "sample_doc.md",  
    ),
    RAGEvalCase(
        "What is dependency injection used for?",
        ["depends", "endpoint"],
        "sample_doc.md",
    ),
    RAGEvalCase(
        "How does the async event loop handle requests?",
        ["event loop", "concurrent"],
        "sample_doc.md",
    ),
    RAGEvalCase("How does Pydantic validation work?",
                ["validation"], "sample_doc.md"),
    RAGEvalCase("What does dependency injection do?",
                ["depends"], "sample_doc.md"),
    RAGEvalCase("How does the async event loop handle requests?",
                ["event loop"], "sample_doc.md"),

    # Code questions over your own chunker.py
    RAGEvalCase("What does fixed_size_chunks do?",
                ["chunks", "tokens"], "app/core/chunker.py"),
    RAGEvalCase("How does recursive_chunks decide where to split?",
                ["separator", "split"], "app/core/chunker.py"),
    RAGEvalCase("How does chunk_python_file handle syntax errors?",
                ["fallback", "recursive"], "app/core/chunker.py"),

    # Mixed / hard
    RAGEvalCase("How are chunks stored in pgvector?",
                ["embedding", "vector"], "app/core/chunker.py"),  # crossover

    # Refusal cases
    RAGEvalCase("What's the boiling point of mercury?", [], "", should_refuse=True),
    RAGEvalCase("Who won the 2024 Super Bowl?", [], "", should_refuse=True),

    RAGEvalCase(
        "What is the recipe for chocolate cake?",
        [],           # no facts expected — this should be refused
        "",           # no source — this shouldn't be in the docs
        should_refuse=True,   # system must say "I don't know"
    ),

]

#------METRIC 1: Fact coverage---------------

def score_fact_coverage(answer: str, facts: list[str])-> float:
    """Returns 0.0 to 1.0 - fraction of expected facts found in anwer."""
    if not facts:
        return 1.0
    
    low = answer.lower()
    return sum(1 for f in facts if f.lower() in low)/len(facts)

#------METRIC 2: Retrieval eval---------------
async def eval_retrieval(cases, top_k=5):
    """Evaluate retrieval quality across all non-refusal cases.
    Returns hit_rate (0.0-1.0) and MRR (0.0-1.0).
    """
    hits = 0
    rr = []
    
    scored = [ c for c in cases if not c.should_refuse]
    for case in scored:
        qv = await rag.embedder.embed_one(case.question)

        results = await rag.store.search(qv, top_k= top_k)

        rank = next(
            (i for i, r in enumerate(results, 1)
            if r.source == case.expected_source),
            None
        )
        if rank: 
            hits += 1
            rr.append(1.0/rank)
        else:
            rr.append(0.0)
    n = len(scored)
    return {
        "hit_rate": hits / n,
        "mrr": sum(rr)/n,
        "n" : n,
    }

#------Metric 3: Faithfulness check----------------
async def eval_faithfulness(answer, chunks) -> FaithfulnessVerdict: 
    """Ask the LLM to judge wheather the answer is supported by the chunks.

    Returns a FaithfulnessVerdict with verdic + explanation.
    """

    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))

    prompt = f"""check if every clain in the answer is supported by the context.

Context:
{context}

Answer:
{answer}

verdict:  supported / partial / unsupported. List unsupported claims."""

    return await rag.llm.complete_structured(prompt, schema=FaithfulnessVerdict)

# -------- Full Answer eval ----------------
async def eval_answers(cases):
    """Run full Rag query for each case. Measure coverage, faithfulness, refusals."""
    coverage_scores = []
    refusal_correct = []
    faith_scores = []

    for case in cases: 
        ans = await rag.query(case.question, top_k=5)

        if case.should_refuse:
            refused = "don't have enough information" in ans.answer.lower()

            refusal_correct.append(refused)
            continue

        coverage_scores.append(score_fact_coverage(ans.answer, case.expected_facts))

        if ans.sources:
            chunk_texts = [s.text for s in ans.sources]

            verdict = await eval_faithfulness(ans.answer, chunk_texts)

            faith_scores.append(
                1.0 if verdict.verdict == "supported"
                else 0.5 if verdict.verdict == "partial"
                else 0.0
            )
    return {
        "avg_fact_coverage": sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0,
        "avg_faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else 0,
        "refusal_accuracy": sum(refusal_correct)/len(refusal_correct) if refusal_correct else 0,

    }

async def main():
    with open("data/sample_doc.md") as f:
        await rag.ingest(f.read(), source="sample_doc.md")


    print("===Retrieval====")
    r = await eval_retrieval(GOLDEN)
    print(f"    hit_rate:  {r['hit_rate']:.2f}    mrr: {r['mrr']:.2f}.   (n={r['n']})")

    print("\n====ANSWERS=====")
    a = await eval_answers(GOLDEN)
    print(f"    fact_coverage: {a['avg_fact_coverage']:.2f}")
    print(f"    faithfulness: {a['avg_faithfulness']:.2f}")
    print(f"    refusal_accuracy: {a['refusal_accuracy']:.2f}")

    await rag.store.delete_by_source("sample_doc.md")
    await rag.store.close()
if __name__ == "__main__":
    asyncio.run(main())
