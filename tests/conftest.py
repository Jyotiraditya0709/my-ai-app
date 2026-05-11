import pytest 
import pytest_asyncio 
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock
from app.main import app

#---------- FIXTURE 1: Client--------------------
@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"

    )as ac: 
        yield ac


#------FIXTURE 2: mock_llm -------------------
@pytest.fixture
def mock_llm(monkeypatch):

    mock = AsyncMock(return_value="Mocked LLM response")

    monkeypatch.setattr("app.core.llm.call_openai", mock)

    monkeypatch.setattr("app.core.llm.call_anthropic", mock)

    return mock


#-------FIXTURE 3: sample_chunks----------
@pytest.fixture
def sample_chunks():
    return [
        {"text":"def auth(user): ...", "source":"auth.py", "score": 0.92},
        {"text": "class User(BaseModel): ...", "source": "models.py", "score":0.87}
    ]