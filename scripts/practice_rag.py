import asyncio
import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

form app.core.rag import rag

async def main():
    #1. Ingest a markdown doc
    print("=== 1. Ingesting simple_doc.md ===")
    with open("data/simple_doc.md") as f:
        text = f.read()

    n = await rag.ingest(text, source="simple_doc.md")
    print(f"Ingested {n} chunks from sample_doc.md")
    
    #2. Ingest a python file( your own chunker. py)
    print("\\n=== 2. Ingesting chunker.py (code) ===")

    with open("app/core/chunker.py") as f:
        code = f.read()
    n_code = await rag.ingest_code(code, file_path="app/core/chunker.py")
    print(f"Ingested {n_code} code chunks from chunker.py")
 
    # 3 . Ask questions
    print("\\n=== 3. Querying ===")
    questions = [
        "How does Pydantic validation work in FastAPI?",
        "What is the dependency injection system used for?",
        "How does the recursive chunker decide where to split?",
        "what is the recipe for a chocolate cake?" # should refuse this 

    ]
    for q in questions:
        print(f"\\nQ:q")
        ans = await rag.query(q, top_k=3)
        print(f"A: {ans.answer}")
        print(f"        ({ans.n_chunks_retrieved} chunks, ${ans.cost_usd: .5f}, {ans.latency_ms}ms)")
        for s in ans.sources:
            print(f"    [{s.rank}] {s.source} {sim={s.similariy:.3f}}")

    #4. cleanup
    print("\\n=== 4. Cleanup ====")
    deleted_md = await rag.store.delete_by_source("sample_doc.md")
    deleted_code = await rag.store.delete_by_source("app/core/chunker.py")
    print(f"Deleted {deleted_md} md chunks, {deleted_code} code chunks")
    await rag.store.close()

asyncio.run(main())