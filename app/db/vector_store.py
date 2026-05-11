# This file is your database layer — the only place in your app that
# talks to PostgreSQL. Everything else imports from here.
#
import asyncpg 
#asyncpg = async postgresql driver 

import os
import json 
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

#------Result Model----------------
@dataclass
class SearchResult:
    id: int
    text: str 
    source: str 
    chunk_index: str 
    similarity: float
    metadata: dict

#------Vector store class------------
class VectorStore:
    def __init__(self):
        self.pool = None
        
        self.db_url = os.getenv("DATABASE_URL")

    async def connect(self):
        """Open the connection pool. Call this once at app startup."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
            )
    
    async def close(self):
        """Close all connection in the pool. Call at app shutdown."""
        if self.pool:
            await self.pool.close()
            self.pool= None
    
    async def _get_pool(self):
        """Get the pool, connecting first if needed."""
        if self.pool is None:
            await self.connect()
        return self.pool
    
    async def insert_chunk(
        self,
        text: str,
        embedding: list[float],
        source: str,
        chunk_index:int,
        metadata: dict = None,
    )-> int:
        """Insert one chunck into the database. Returns the noew row's ID."""
        pool = await self._get_pool()

        embedding_str = "[" + ",".join(str(x) for x in embedding)+"]"

        metadata_json = json.dumps(metadata or {})

        async with pool.acquire() as conn:
            #pool.acquire = borrow a connection from the pool
            # async with = return it automatically when done

            row = await conn.fetchrow(
                """
                INSERT INTO chunks ( text, embedding, source, chunk_index, metadata)
                VALUES ($1, $2::vector, $3, $4, $5)
                RETURNING id
                """,
                text,
                embedding_str,
                source,
                chunk_index,
                metadata_json,
            )
        return row["id"]

    async def insert_chunks(self, chunks: list[dict]) -> int:
        """Insert multiple chunks efficiently. Returns count of inserted rows.
        chunks = list of dicts, each with keys:
        text, embedding, source, chunk_index, metadata(optional)
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO chunks (text, embedding, source, chunk_index, metadata)
                VALUES ($1, $2::vector, $3, $4, $5)
                """,
                [
                    (
                        c["text"],
                        "[" + ",".join(str(x) for x in c["embedding"]) + "]",
                        c["source"],
                        c["chunk_index"],
                        json.dumps(c.get("metadata", {})),

                    )
                    for c in chunks
                ]
            )
        return len(chunks)
    
    async def search(self, query_embedding:list[float],top_k: int = 5, source_filter:Optional[str] = None,)->list[SearchResult]:
        """Find the top-K most similar chubks to a query embedding.

        This is the core of RAG retrueval - given a question embedding,
        find the most relevant chunks to include in the LLM prompt.
        """
        pool = await self._get_pool()

        query_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        async with pool.acquire() as conn:
            if source_filter:
                rows = await conn.fetch(
                    """SELECT
                         id, text, source, chunk_index, metadata,
                         1 - (embedding <=> $1::vector) AS similarity
                    FROM chunks
                    WHERE source = $2
                    ORDER BY embedding <=> $1:: vector
                    LIMIT $3
                    """,
                    query_str,
                    source_filter,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT 
                        id, text, source, chunk_index, metadata,
                        1 - (embedding <=> $1::vector) AS similarity
                    FROM chunks
                    ORDER BY embedding <=> $1:: vector
                    LIMIT $2
                    """,
                    query_str,
                    top_k,
                )
        return [
            SearchResult(
                id= row["id"],
                text=row["text"],
                source=row["source"],
                chunk_index=row["chunk_index"],
                similarity=float(row["similarity"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},

            )
            for row in rows
        ]

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source file. Returns count deleted."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chunks WHERE source = $1",
                source,
            )
            return int(result.split()[-1])

    async def count(self) -> int:
            """Count total chunks in the database."""
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                return await conn.fetchval("SELECT COUNT(*) FROM chunks")

#---------SINGLETON--------------------
vector_store= VectorStore()
