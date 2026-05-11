import pytest
from app.core.embeddings import embedder, cosine_similarity, find_most_similar

#-------UNIT TESTS (no api calls) --------------
#these test the MATH functions with fake vectors - fast, free, no network needed.

def test_cosine_similarity_identical():
    a = [1.0,0.0,0.0]
    b = [1.0, 0.0,0.0]
    assert cosine_similarity(a,b) == pytest.approx(1.0)

def test_cosine_similarity_orthogonal():
    # Orthogonal (perpendicular) vectors = completely unrelated = 0.0
    a = [1.0,0.0, 0.0]
    b = [0.0,1.0, 0.0]
    assert cosine_similarity(a,b) == pytest.approx(0.0)

def test_find_most_similar_returns_top_k():
    query = [1.0,0.0,0.0]
    candidates = [
               [1.0, 0.0, 0.0],   # index 0 — identical to query, similarity=1.0
        [0.0, 1.0, 0.0],   # index 1 — orthogonal, similarity=0.0
        [0.9, 0.1, 0.0],   # index 2 — very similar, similarity~0.99 
    ]
    results = find_most_similar(query, candidates, top_k=2)
    assert len(results) == 2
    assert results[0][0] == 0

# ─── INTEGRATION TESTS (hit real OpenAI API) ─────────────────────────────────
# These test actual embeddings — skip them in fast test runs with:
# pytest -m "not integration"

@pytest.mark.integration # mark == label so you can filted with -m flag
async def test_embed_one_real_call():
    """Real API call - varifies the embedding model returns the right shape."""
    vec = await embedder.embed_one("hello")
    assert len(vec) == 1536
    assert all(isinstance(x, float) for x in vec)

@pytest.mark.integration
async def test_embed_many_preserves_order():
    """Verify that vectors come back in the same order as input texts."""
    texts = ["first", "second", "third"]
    vectors = await embedder.embed_many(texts)

    assert len(vectors) == 3
    first_again = await embedder.embed_one("first")
    assert cosine_similarity(vectors[0], first_again) > 0.99