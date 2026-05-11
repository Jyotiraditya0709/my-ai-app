import pytest
from app.db.vector_store import VectorStore, SearchResult

@pytest.fixture
async def store():
    s = VectorStore()
    yield s 
    await s.close()

@pytest.mark.integration
async def test_insert_and_search(store):
    fake_vec=[0.0] * 1536
    fake_vec[0] =1.0

    await store.insert_chunk(
        text="test chunk",
        embedding=fake_vec,
        source="test_file.py",
        chunk_index=0,
    )
    results = await store.search(fake_vec, top_k =1)
    assert len(results) == 1
    assert results[0].text == "test chunk"
    assert results[0].similarity > 0.99

    await store.delete_by_source("test_file.py")

@pytest.mark.integration
async def test_filtered_search(store):
    vec = [0.0] * 1536
    vec[0] = 1.0

    await store.insert_chunk(text="A", embedding=vec, source="file_a.py",chunk_index=0)
    await store.insert_chunk(text="B", embedding=vec, source="file_b.py", chunk_index=0)

    results_a = await store.search(vec, top_k=5, source_filter="file_a.py")
    assert all(r.source == "file_a.py" for r in results_a)

    await store.delete_by_source("file_a.py")
    await store.delete_by_source("file_b.py")