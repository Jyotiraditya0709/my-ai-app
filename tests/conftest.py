import pytest 
import pytest_asyncio 
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock
from app.main import app
from unittest.mock import AsyncMock, MagicMock 

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
    from app.core.llm import LLMClient

    llm_response = MagicMock()
    llm_response.content = "Mocked LLM response"
    llm_response.cost_usd = 0.0001

    mock = AsyncMock(return_value=llm_response)
    monkeypatch.setattr(LLMClient, "complete", mock)
    return mock


#-------FIXTURE 3: sample_chunks----------
@pytest.fixture
def sample_chunks():
    return [
        {"text":"def auth(user): ...", "source":"auth.py", "score": 0.92},
        {"text": "class User(BaseModel): ...", "source": "models.py", "score":0.87}
    ]