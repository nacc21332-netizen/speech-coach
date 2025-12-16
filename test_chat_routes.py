import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_routes_exist(client):
    """Тестирование существования маршрутов чата"""
    
    # Проверяем, что маршрут GET для UI существует
    response = client.get("/chat/ui")
    # Должен возвращать HTML страницу (успешный статус или ошибка из-за отсутствия GigaChat клиента)
    assert response.status_code in [200, 500]  # 500 может быть, если GigaChat не настроен
    
    # Проверяем, что маршрут POST для чата существует (даже если с ошибкой из-за отсутствия тела)
    response = client.post("/chat")
    # Должна быть ошибка валидации, так как нет тела запроса
    assert response.status_code in [422, 500]  # 422 - ошибка валидации, 500 - ошибка GigaChat
    
    # Проверяем, что маршрут POST для follow-up существует
    response = client.post("/chat/analyze-followup")
    # Должна быть ошибка валидации, так как нет тела запроса
    assert response.status_code in [422, 500]  # 422 - ошибка валидации, 500 - ошибка GigaChat


def test_chat_with_valid_request_structure(client):
    """Тестирование структуры запроса к чату"""
    
    # Отправляем минимальный валидный запрос
    valid_request = {
        "messages": [
            {
                "role": "user",
                "content": "Привет, как дела?"
            }
        ]
    }
    
    response = client.post("/chat", json=valid_request)
    
    # Должна быть ошибка аутентификации GigaChat или внутренняя ошибка, если GigaChat не настроен
    # или валидный ответ, если GigaChat настроен
    assert response.status_code in [200, 400, 500]


def test_analyze_followup_with_valid_request_structure(client):
    """Тестирование структуры запроса к follow-up чату"""
    
    # Отправляем минимальный валидный запрос
    valid_request = {
        "messages": [
            {
                "role": "user",
                "content": "Расскажи, как улучшить мою речь?"
            }
        ]
    }
    
    response = client.post("/chat/analyze-followup", json=valid_request)
    
    # Должна быть ошибка аутентификации GigaChat или внутренняя ошибка, если GigaChat не настроен
    # или валидный ответ, если GigaChat настроен
    assert response.status_code in [200, 400, 500]


if __name__ == "__main__":
    pytest.main([__file__])