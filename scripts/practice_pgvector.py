import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.embeddings import embedder
from app.db.vector_store import vector_store

async def main():
    #-------1. Insert---------------------------
    print("=== 1. Insert chunks ===")

    texts =[
        "FastAPI uses async/await for non blocking endpoints.",
        "Pydantic models validate request and response data.",
        "Authentication in FastAPI is typically done with OAuth2 dependencies.",
        "Pgvector adds a VECTOR data type to PostgreSQL.",
        "Cosine similarity measures the angle between two vectors.",
    ]

    vectors = await embedder.embed_many(texts)

    chunks = [
        {
            "text":text,
            "embedding":vec,
            "source":"fastapi_notes.md",
            "chunk_index":i,
        }
        for i,(text, vec) in enumerate(zip(texts, vectors))

    ]
    inserted = await vector_store.insert_chunks(chunks)
    print(f"Inserted {inserted} chunks")

    total = await vector_store.count()
    print(f"Total chunks in DB: {total}")
    print()

    #----------2. SEARCH-----------------
    print("=== 2. Vector search ===")

    queries = [
        "How do I add authentication?",    # should match OAuth2 chunk
        "What is a vector database?",       # should match pgvector chunk
        "How does data validation work?",   # should match Pydantic chunk
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        query_vec = await embedder.embed_one(query)

        results = await vector_store.search(query_vec, top_k= 2)
        for r in results:
            print(f" Similarity={r.similarity:.3f} {r.text}")

     # ─── 3. FILTERED SEARCH ──────────────────────────────────────
    print("\n=== 3. Filtered by source ===")

    query_vec = await embedder.embed_one("authentication")       
    results = await vector_store.search(
        query_vec,
        top_k = 3, 
        source_filter = "fastapi_notes.md"

    )
    for r in results:
        print(f" {r.similarity:.3f} {r.text}")

     # ─── 4. CLEANUP ──────────────────────────────────────────────
    print("\n=== 4. Cleanup ===")
    
    deleted = await vector_store.delete_by_source("fastapi_notes.md")
    print(f"Deleted {deleted} chunks")

    await vector_store.close()

asyncio.run(main())