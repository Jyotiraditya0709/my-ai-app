import pytest

async def test_health_endpoints(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code== 200
    assert "status" in response.json()

async def test_query_valid_request(client, mock_llm):
    response = await client.post("/query", json={
        "question": "What is RAG?",
        "top_k" : 3
    })
    assert response.status_code == 200
    assert "question" in response.json()

async def test_query_empty_question_returns_422(client):
    response = await client.post("/query", json={
        "question":"",
        "top_k": 3
    })
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("question" in str(e) for e in errors)

async def test_query_top_k_out_of_range_returns_422(client):
    response = await client.post("/query", json={
        "question": "Valid question here",
        "top_k" : 999
    })
    assert response.status_code == 422

async def test_get_item_valid_id(client):
    response = await client.get("/items/42")
    assert response.status_code == 200
    assert response.json()["item_id"] == 42

async def test_get_item_invalid_id_returns_404(client):
    response = await client.get("/items/-1")
    assert response.status_code == 404

async def test_echo_repeats_correctly(client):
    response = await client.post("/echo", json={
        "text": "hello",
        "repeat": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert data["repeated"] == ["hello", "hello", "hello"]
    assert data["char_count"] == 5

async def test_search_returns_results(client):
    response = await client.get("/search?q=authentication&limit=3")
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert data["query"] == "authentication"
