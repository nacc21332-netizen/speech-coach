import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.api.routes.chat import ChatRequest, ChatMessage


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint_exists(client):
    """Тест проверяет, что эндпоинт чата существует"""
    with patch('app.api.deps.get_gigachat_client') as mock_get_client:
        # Создаем мок клиента
        mock_client = AsyncMock()
        mock_client._access_token = "mock_token"
        mock_client.api_url = "https://mock-api.com"
        mock_client.model = "mock-model"
        mock_client.max_tokens = 1000
        
        # Настройка мока аутентификации
        mock_client.authenticate = AsyncMock()
        
        mock_get_client.return_value = mock_client
        
        # Подготавливаем данные запроса
        request_data = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Привет!")
            ]
        ).dict()
        
        # Проверяем, что эндпоинт отвечает
        response = client.post("/chat", json=request_data)
        
        # Ожидаем ошибку аутентификации или другую ошибку, но не 404
        assert response.status_code != 404


@pytest.mark.asyncio
async def test_analyze_followup_endpoint_exists(client):
    """Тест проверяет, что эндпоинт follow-up чата существует"""
    with patch('app.api.deps.get_gigachat_client') as mock_get_client:
        # Создаем мок клиента
        mock_client = AsyncMock()
        mock_client._access_token = "mock_token"
        mock_client.api_url = "https://mock-api.com"
        mock_client.model = "mock-model"
        mock_client.max_tokens = 1000
        
        # Настройка мока аутентификации
        mock_client.authenticate = AsyncMock()
        
        mock_get_client.return_value = mock_client
        
        # Подготавливаем данные запроса
        request_data = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Расскажи о моем анализе")
            ],
            analysis_id="test-analysis-id"
        ).dict()
        
        # Проверяем, что эндпоинт отвечает
        response = client.post("/chat/analyze-followup", json=request_data)
        
        # Ожидаем ошибку аутентификации или другую ошибку, но не 404
        assert response.status_code != 404


def test_chat_ui_endpoint(client):
    """Тест проверяет, что UI эндпоинт чата существует"""
    response = client.get("/chat/ui")
    
    # Проверяем, что эндпоинт существует и возвращает HTML
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]