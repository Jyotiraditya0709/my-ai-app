from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from app.core.llm import llm

app = FastAPI(title="Day 3 AI API", version="0.1.0")

#---Models---
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k : int = Field(default=5, ge=1, le=20)
    model: str = "gpt-4"

class EchoRequest(BaseModel):
    text: str
    repeat: int = Field(default=1, ge=1, le=5)

#---Routes---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Day 3 FastAPI server running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/items/{item_id}")
async def get_item(item_id: int, include_meta: bool = False):
    if item_id < 0:
        raise HTTPException(status_code=404, detail="Item not found")
    result = {"item_id": item_id, "name": f"Item {item_id}"}
    if include_meta:
        result["meta"] = {"created":"2026-03-14"}
    return result

@app.post("/echo")
async def echo(req: EchoRequest):
    return {
        "original":req.text,
        "repeated": [req.text]*req.repeat,
        "char_count": len(req.text)

    }

@app.post("/query")
async def query(req: QueryRequest):
    return{
        "question":req.question,
        "anwser":f"[Placeeholder]Answer to: {req.question}",
        "top_k_used": req.top_k,
        "model": req.model,
        "sources": ["doc1.py", "doc2.py"]

    }

@app.get("/search")
async def search(q:str, limit:int=10, model:Optional[str] =None):
    return {
        "query":q,
        "limit": limit,
        "results":[f"Result {i} for {q}" for i in range(limit)]
    }





# ---Streaming---
@app.post("/stream")
async def stream(req: QueryRequest):
    async def generate():
        s = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": req.question}],
            stream=True,
        )
        async for chunk in s:
            token = chunk.choices[0].delta.content
            if token:
                yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")