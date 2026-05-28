import pytest
from unittest.mock import AsyncMock, MagicMock
from app.core.rag import RAG, RAGAnswer
from app.db.vector_store import SearchResult

@pytest.fixture
def mock_components():
    """Build a RAG with all components mocked — pure unit test, no DB or API."""
    embedder = MagicMock()
    embedder.embed_one = AsyncMock(return_value=[0.1] * 1536)
    embedder.embed_many = AsyncMock(return_value=[[0.1] * 1536] * 2)

    store = MagicMock()
    store.delete_by_source = AsyncMock(return_value=0)
    store.insert_chunks = AsyncMock(return_value=2)
    store.search = AsyncMock(return_value=[
        SearchResult(id=1, text="chunk 1", source="test.md",
                     chunk_index=0, similarity=0.9),
        SearchResult(id=2, text="chunk 2", source="test.md",
                     chunk_index=1, similarity=0.8),
    ])

    llm_response = MagicMock()
    llm_response.content = "Mocked answer with [1] citation."
    llm_response.cost_usd = 0.0001

    llm = MagicMock()
    llm.complete = AsyncMock(return_value=llm_response)

    return RAG(embedder=embedder, store=store, llm=llm)

async def test_query_returns_typed_answer(mock_components):
    rag = mock_components
    answer = await rag.query("test question", top_k=3)

    assert isinstance(answer, RAGAnswer)
    assert answer.answer == "Mocked answer with [1] citation."
    assert len(answer.sources) == 2
    assert answer.sources[0].rank == 1
    assert answer.cost_usd > 0

async def test_query_with_no_chunks_refuses(mock_components):
    rag = mock_components
    rag.store.search = AsyncMock(return_value=[])

    answer = await rag.query("test question")
    assert "don't have enough information" in answer.answer.lower()
    assert answer.sources == []
    assert answer.cost_usd == 0.0

async def test_query_filters_low_similarity(mock_components):
    rag = mock_components
    # All results below threshold
    rag.store.search = AsyncMock(return_value=[
        SearchResult(id=1, text="x", source="a", chunk_index=0, similarity=0.2),
    ])
    rag.min_similarity = 0.3

    answer = await rag.query("test")
    assert "don't have enough information" in answer.answer.lower()