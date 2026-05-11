import pytest
from app.core.chunker import fixed_size_chunks, recursive_chunks, chunk_python_file

def test_fixed_size_no_overlap():
    text = "Hello" * 1000
    chunks = fixed_size_chunks(text, chunk_tokens = 100, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)

def test_fixed_size_with_overlap():
    text = "hello " * 1000
    chunks = fixed_size_chunks(text, chunk_tokens = 100, overlap=20)
    assert len(chunks) > 1

def test_short_text_one_chunks():
    text = "short"
    assert fixed_size_chunks(text, 300) == [text]

def test_recursive_respects_paragraphs():
    text = "Para one. \n\n two. \n\nPara three."
    chunks = recursive_chunks(text, chunk_size=15)
    assert len(chunks) >= 2

def test_python_chunking_finds_functions():
    code ='''
import os 

def first():
    return 1

def second():
    return 2

class Thing:
    pass

''' 
    chunks = chunk_python_file(code,"test.py")
    kinds = [c["kind"] for c in chunks]
    assert "imports" in kinds     # should find the import
    assert "function" in kinds    # should find the functions
    assert "class" in kinds   

def test_python_chunking_handles_syntax_error():
    # Broken code should NOT crash — falls back to recursive chunking
    code = "def broken(:\n    return 1" 
    chunks = chunk_python_file(code, "test.py")
    assert len(chunks) >= 1   # at least one chunk returned, no crash