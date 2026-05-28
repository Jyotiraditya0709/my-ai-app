from __future__ import annotations
from typing import Callable
from pydantic import BaseModel
import time

from app.core.embeddings import embedder as default_embedder, Embedder
from app.core.chunker import recursive_chunks, chunk_python_file
from app.core.llm import llm as default_llm, LLMClient 
from app.core.prompts import rag_with_citations
from app.db.vector_store import vector_store as default_store, VectorStore, SearchResult


class RAGSource(BaseModel):
    rank: int
    text: str
    source: str
    chunk_index: int
    similarity: float


class RAGAnswer(BaseModel):
    answer: str
    sources: list[RAGSource]
    n_chunks_retrieved: int
    cost_usd: float
    latency_ms: int


class RAG:
    def __init__(
        self,
        embedder: Embedder = default_embedder,
        store: VectorStore = default_store,
        llm: LLMClient = default_llm,
        min_similarity: float = 0.3,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.min_similarity = min_similarity

    async def ingest(
        self,
        text: str,
        source: str,
        chunker: Callable[[str], list[str]] = recursive_chunks,
        replace: bool = True,
    ) -> int:
        """Ingest a prose/markdown document. Returns number of chunks stored."""
        if replace:
            await self.store.delete_by_source(source)

        chunks = chunker(text)
        if not chunks:
            return 0

        vectors = await self.embedder.embed_many(chunks)

        rows = [
            {
                "text": chunk,
                "embedding": vec,
                "source": source,
                "chunk_index": i,
                "metadata": {},
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        await self.store.insert_chunks(rows)
        return len(rows)

    async def ingest_code(
        self,
        source_code: str,
        file_path: str,
        replace: bool = True,
    ) -> int:
        """Ingest a Python file using AST-based chunking."""
        if replace:
            await self.store.delete_by_source(file_path)

        code_chunks = chunk_python_file(source_code, file_path)
        if not code_chunks:
            return 0

        texts = [c["text"] for c in code_chunks]
        vectors = await self.embedder.embed_many(texts)

        rows = [
            {
                "text": c["text"],
                "embedding": v,
                "source": c["source"],
                "chunk_index": i,
                "metadata": {
                    "kind": c["kind"],
                    "name": c["name"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                },
            }
            for i, (c, v) in enumerate(zip(code_chunks, vectors))
                    ]
        await self.store.insert_chunks(rows)
        return len(rows)

    async def query(
        self,
        question: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> RAGAnswer:
        """Answer a question using retrieved context."""
        start = time.time()

        # 1. Embed the question
        query_vec = await self.embedder.embed_one(question)

        # 2. Retrieve top-K chunks
        results = await self.store.search(
            query_vec,
            top_k=top_k,
            source_filter=source_filter,
        )

        # 3. Filter low-similarity results
        good_results = [r for r in results if r.similarity >= self.min_similarity]

        # 4. Handle no-context case
        if not good_results:
            return RAGAnswer(
                answer="I don't have enough information in the indexed content to answer that question.",
                sources=[],
                n_chunks_retrieved=len(results),
                cost_usd=0.0,
                latency_ms=round((time.time() - start) * 1000),
            )

        # 5. Build the prompt with citations
        prompt = rag_with_citations(question, good_results)

        # 6. Call the LLM
        response = await self.llm.complete(
            prompt,
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=500,
        )

        # 7. Build the typed response
        sources = [
            RAGSource(
                rank=i + 1,
                text=r.text,
                source=r.source,
                chunk_index=r.chunk_index,
                similarity=r.similarity,
            )
            for i, r in enumerate(good_results)
        ]

        return RAGAnswer(
            answer=response.content,
            sources=sources,
            n_chunks_retrieved=len(good_results),
            cost_usd=response.cost_usd,
            latency_ms=round((time.time() - start) * 1000),
        )


rag = RAG()
