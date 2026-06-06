from __future__ import annotations

from app.core.embeddings import embedder
from app.db.vector_store import vector_store, SearchResult
import asyncpg, json

# Lazy reranker loading
_reranker = None

def _load_reranker():
    """Load the cross-encoder model the first time it's needed."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# Reciprocal rank fusion 
def reciprocal_rank_fusion(
        semantic: list[SearchResult],
        keyword: list[SearchResult],
        k: int = 60,
        top_k: int = 5,

)-> list[SearchResult]:
    scores: dict[int, float] = {}
    by_id: dict[int, SearchResult] = {}
    for rank, r in enumerate(semantic, start=1):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0/ (k + rank)
        by_id[r.id] = r

    for rank, r in enumerate(keyword, start=1):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0/ (k + rank)
        by_id.setdefault(r.id, r)

    
    ranked = sorted(scores.items(), key= lambda x: x[1], reverse= True)

    return [by_id[rid] for rid, _ in ranked[:top_k]]

# keyword search (bm25 like)
async def keyword_search(query_text: str, top_k: int = 20) -> list[SearchResult]:
    """BM25 like search using postgres full text search."""
    pool = await vector_store._get_pool() # get the conn pool from your existing vec-store
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, text, source, chunk_index, metadata,
                ts_rank(ts,plainto_tsquery('english', $1)) AS score
            FROM chunks
            WHERE ts @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
            """,
            query_text, top_k,
        )
    return [
        SearchResult(
            id=r["id"],
            text=r["text"], 
            source=r["source"],
            chunk_index=r["chunk_index"],
            similarity=float(r["score"]),
            metadata=json.loads(r["metadata"]) if r["metadata"] else {}

            
        )
        for r in rows
    ]

async def hybrid_search(
        query_text: str,
        top_k: int = 5,
        candidate_pool: int = 20,

)-> list[SearchResult]:
    """Semantic + BM25, fused by RRF."""

    query_vec = await embedder.embed_one(query_text)
    semantic = await vector_store.search(query_vec, top_k= candidate_pool)

    keyword = await keyword_search(query_text, top_k=candidate_pool)
    return reciprocal_rank_fusion(semantic, keyword, top_k=top_k)

#hybrid _ rerank
async def hybrid_with_rerank(
        query_text : str,
        top_k : int = 5,
        candidate_pool: int =20,

)-> list[SearchResult]:
    """Hybrid retrieal, then cross-encoder rerank top candidates."""

    # step 1: get hybrid candidtes
    candidates = await hybrid_search(
        query_text, top_k=candidate_pool, candidate_pool=candidate_pool

    )
    if not candidates: 
        return []
    
    #step 2: load the reranker
    reranker = _load_reranker()

    #step 3 : create (question, passage) pairs fro the cross encoder
    pairs = [(query_text, c.text) for c in candidates]

    #step 4: score all pairs
    scores = reranker.predict(pairs)

    #step 5: sort by reranker
    scored = sorted(zip(candidates, scores), key= lambda x: x[1], reverse= True)

    return [c for c, _ in scored[:top_k]]


