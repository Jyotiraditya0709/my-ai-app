import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.chunker import (
    fixed_size_chunks,
    recursive_chunks,
    semantic_chunks,
    chunk_python_file,
)

import tiktoken
ENC = tiktoken.encoding_for_model("gpt-4")

def stats(name: str, chunks:list[str]):
    """Print statistics(measurements) for a list of chunks.

    Shows how many chunks, average/min/max token count, preview of first chunks.
    """
    if not chunks:
        print(f"{name}: 0 chunks")
        return 

    counts = [len(ENC.encode(c)) for c in chunks]

    print(f"\n{name}")
    print(f"  chunks: {len(chunks)}")
    print(f"  tokens: avg={sum(counts)/len(counts):.0f}  "
        f"min={min(counts)}  max={max(counts)}")
    print(f"  first chunk preview: {chunks[0][:120]}...")

async def main():
    with open("data/sample_doc.md") as f:
        text = f.read()

    print("=== Comparing chunking strategies on sample_doc.md ===")

    #---Strategy 1 : fixed size --------------
    fixed = fixed_size_chunks(text, chunk_tokens=150, overlap=20)   
    stats("Strategy 1: Fixed-size(150 tokens, 20overlap)", fixed)

    # ── Strategy 2: Recursive character ──────────────────────────
    # Splits on paragraph breaks first, then smaller boundaries
    recursive = recursive_chunks(text, chunk_size=600, overlap=80)
    stats("Strategy 2: Recursive character (600 chars, 80 overlap)", recursive)

    # ── Strategy 3: Semantic ──────────────────────────────────────
    # Embeds sentences and splits where topic changes
    # This makes a real API call — requires OpenAI credits
    semantic = await semantic_chunks(text, similarity_threshold=0.75)
    stats("Strategy 3: Semantic (threshold 0.75)", semantic)

    # ── Strategy 4: Code-aware ────────────────────────────────────
    print("\n\n=== Code chunking ===")
    with open("app/core/chunker.py") as f:
        code = f.read()
    
    code_chunks = chunk_python_file(code, "app/core/chunker.py")
    
    print(f"Found {len(code_chunks)} chunks in chunker.py:")
    for c in code_chunks:
        print(f"  - {c['kind']}: {c['name']} "
              f"(lines {c['start_line']}-{c['end_line']})")
        # shows: "- function: fixed_size_chunks (lines 14-32)"


asyncio.run(main())