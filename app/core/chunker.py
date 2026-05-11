from __future__ import annotations

import ast
import re 
import tiktoken 

ENCODER = tiktoken.encoding_for_model("gpt-4")


# ─── STRATEGY 1: FIXED-SIZE CHUNKS ───────────────────────────────────────────
def fixed_size_chunks(
    text: str,
    chunk_tokens:int = 300,
    overlap:int = 50,
) -> list[str]:
    """Split text into chunks of exactly N tokens with overlap."""

    tokens = ENCODER.encode(text)

    if len(tokens) <= chunk_tokens:
        return [text]

    chunks = []
    step = chunk_tokens - overlap


    for i in range(0, len(tokens),step):
        chunk_tok = tokens[i: i + chunk_tokens]
        chunks.append(ENCODER.decode(chunk_tok))

        if i + chunk_tokens >= len(tokens):
            break

    return chunks

# ─── STRATEGY 2: RECURSIVE CHARACTER CHUNKS ──────────────────────────────────
def recursive_chunks(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
    separators : list[str] | None = None,
)-> list[str]:
    """Recursive character chunking - respects natural boundaries."""
    
    if separators is None:
        separators = ["\n\n", "\n", ".", " ", ""]

    if len(text) <= chunk_size:
        return [text]
    
    sep = separators[0]

    if sep == "":
        return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    parts = text.split(sep)
    chunks : list[str] = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
    
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            if len(part) > chunk_size:
                chunks.extend(
                    recursive_chunks(part, chunk_size, overlap, separators[1:])
                )
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)

    return chunks

# ─── STRATEGY 3: SEMANTIC CHUNKS ────────────────────────────────────────────
async def semantic_chunks(
    text: str,
    similarity_threshold: float = 0.75,

)->list[str]:
    """Split at semantic boundaries detected via snetence-level embedding."""
    
    from app.core.embeddings import embedder, cosine_similarity

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]

    if len(sentences) <= 1:
        return sentences

    sentences_vecs = await embedder.embed_many(sentences)

    chunks: list[str] = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(sentences_vecs[i - 1], sentences_vecs[i])

        if sim < similarity_threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current: 
        chunks.append(" ".join(current))
    
    return chunks

# ─── STRATEGY 4: CODE-AWARE PYTHON CHUNKING ──────────────────────────────────
def chunk_python_file(source: str, file_path: str) -> list[dict]:
    """Split a Python file at function/class boundaries using AST."""

    try: 
        tree = ast.parse(source)

    except SyntaxError: 
        return [
            {
                "text": c, 
                "kind": "raw",      # kind = what type of chunk this is
                "name": None,       # no function/class name for raw chunks
                "start_line": None, 
                "end_line": None, 
                "source": file_path
            }
            for c in recursive_chunks(source)
        ]   
    lines = source.split("\n")
    chunks : list[dict] = []

    import_lines = [
        n for n in tree.body
        if isinstance(n,(ast.Import, ast.ImportFrom))

    ]
    if import_lines:
        first = import_lines[0].lineno - 1
        last = import_lines[-1].end_lineno 
        chunks.append({
            "text" : "\n".join(lines[first:last]),
            "kind":"imports",
            "name": None,
            "start_line" : first + 1,
            "end_line" : last,
            "source": file_path,
        })

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno

            chunks.append({
                "text": "\n".join(lines[start:end]),
                "kind": type(node).__name__.replace("Def", "").lower(),
                "name" : node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "source": file_path,
            })

    return chunks