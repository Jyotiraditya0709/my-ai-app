from fastapi import APIRouter, HTTPException

from pydantic import BaseModel, Field
from app.core.rag import rag, RAGAnswer

router = APIRouter(prefix="/api", tags=["query"])

class QueryRequest(BaseModel):
    """Request body(what the client sends) for /api/query."""
    question: str = Field(
        min_length = 3,
        max_length = 500,
        description="Natural-language question about the codebase."

    )
    top_k:int= Field(
        default=5,
        ge=1,
        le=10,
        description="Number of chunks retrieve."
    )
    source_filter:str | None = Field(
        default=None,
        description="Restrict to one source file(otpional)."
    )

@router.post("/query", response_model=RAGAnswer)
async def query(req: QueryRequest)-> RAGAnswer:
    """Ask a question. Returns a cited answer or 'I don't know'."""
    try:
        return await rag.query(
            question=req.question,
            top_k=req.top_k,
            source_filter=req.source_filter,
        )
    except Exception as e: 
        raise HTTPExecption(
            status_code=500,
            detail="Query failed. Please try again.",
        ) from e

@router.get("/health")
async def health()-> dict:
    """Liveness check - used by Render to know if the service is running."""
    return {"status":"ok"}

@router.get("/stats")
async def stats() -> dict:
    """Quick visibility (insight) into what's indexed."""
    total = await rag.store.count()
    return {"total_chunks":total}
