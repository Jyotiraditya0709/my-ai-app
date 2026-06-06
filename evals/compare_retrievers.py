import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.rag import rag
from app.core.retriever import hybrid_search, hybrid_with_rerank
from evals.rag_eval import GOLDEN

async def measure(name, retrieve_fn, cases, top_k=5 ):
    """Run one retrieval strategy against all cases and return metrics."""
    hits, rr = 0, []
    scored = [c for c in cases if not c.should_refuse]

    for case in scored: 
        results = await retrieve_fn(case.question, top_k=top_k)

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
        "name": name,
        "hit_rate": hits/n,
        "mrr": sum(rr)/n,
    }

async def main():
    # ingest doc
    with open("data/sample_doc.md") as f:
        await rag.ingest(f.read(), source="sample_doc.md")
    with open("app/core/chunker.py") as f:
        await rag.ingest_code(f.read(), file_path="app/core/chunker.py")

        async def baseline(q, top_k):
            qv = await rag.embedder.embed_one(q)
            return await rag.store.search(qv, top_k=top_k)
        
        results = []
        results .append(await measure("Baseline(semantic)", baseline, GOLDEN))
        results.append(await measure("Hybrid (BM25 + vec)", hybrid_search, GOLDEN))
        results.append(await measure("Hybrid + Rerank", hybrid_with_rerank, GOLDEN))

        # print the comparison tale
        print(f"\n{'name':<28} {'hit_rate':>10} {'mrr':>8}")
        print("-" * 50)
        for r in results:
            print(f"{r['name']:<28} {r['hit_rate']:>10.2f} {r['mrr']:>8.2f}")

        # ── Cleanup ───────────────────────────────────────────────────
        await rag.store.delete_by_source("sample_doc.md")
        await rag.store.delete_by_source("app/core/chunker.py")
        await rag.store.close()
asyncio.run(main())


            