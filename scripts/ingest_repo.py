"""Ingest a local repo into pgvector.

Usage: python scripts/ingest_repo.py/ingest_repo.py /path/to/repo/fastapi
"""

import asyncio, sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.rag import rag
SKIP_DIRS = {"__pycache__", ".git", "tests", "test", "docs", ".venv", "node_modules", ".agents"}
SKIP_FILES = {"__init__.py"}
MAX_FILE_BYTES = 50_000

def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if not path.is_file():           # ← NEW: skip directories
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        files.append(path)
    return sorted(files)

async def main(repo_root: str, repo_name: str):
    root = Path(repo_root)
    files = find_python_files(root)
    print(f"Found {len(files)} Python files to ingest from {root}")

    total_chunks = 0
    for i, file in enumerate(files, 1):
        rel = file.relative_to(root)
        source_path = f"{repo_name}/{rel}"
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[{i}/{len(files)}] SKIP (encoding): {rel}")
            continue

        n = await rag.ingest_code(text, file_path=source_path)
        total_chunks += n
        print(f"[{i}/{len(files)}] {n:3d} chunks: {rel}")

    print(f"\nDone. {total_chunks} total chunks indexed.")
    await rag.store.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_repo.py /path/to/repo [repo_name]")
        sys.exit(1)
    repo_root = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else Path(repo_root).name
    asyncio.run(main(repo_root, repo_name))


