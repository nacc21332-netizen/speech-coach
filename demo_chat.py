#!/usr/bin/env python3
"""
Демонстрационный скрипт для показа работы чата с GigaChat
"""

import asyncio
from unittest.mock import AsyncMock, patch
from app.api.routes.chat import ChatRequest, ChatMessage
from app.services.gigachat import GigaChatClient


async def demonstrate_chat_functionality():
    """
    Демонстрирует, как будет работать чат с GigaChat при наличии API ключа
    """
    print("Демонстрация работы чата с GigaChat")
    print("="*50)
    
    # Создаем мок клиента для демонстрации
    mock_client = AsyncMock()
    mock_client._access_token = "demo_token_12345"
    mock_client.api_url = "https://demo-gigachat-api.com"
    mock_client.model = "gigachat:latest"
    mock_client.max_tokens = 1000
    mock_client.authenticate = AsyncMock()
    mock_client.client = AsyncMock()
    
    # Настройка мока для имитации ответа от GigaChat
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Привет! Я GigaChat, и я готов помочь вам улучшить навыки публичных выступлений."
                }
            }
        ],
        "usage": {
            "total_tokens": 15
        }
    }
    mock_client.client.post = AsyncMock(return_value=mock_response)
    
    print("1. Имитация аутентификации...")
    await mock_client.authenticate()
    print(f"   Токен аутентификации: {mock_client._access_token[:10]}...")
    
    print("\n2. Подготовка сообщений для отправки...")
    chat_request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="Привет! Как мне улучшить свои навыки публичных выступлений?")
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    print(f"   Сообщений: {len(chat_request.messages)}")
    print(f"   Температура: {chat_request.temperature}")
    print(f"   Макс. токенов: {chat_request.max_tokens}")
    
    print("\n3. Отправка запроса в GigaChat...")
    # Имитация вызова чат-эндпоинта
    formatted_messages = []
    
    # Добавим системное сообщение
    system_message = {
        "role": "system",
        "content": """Ты - помощник по развитию навыков публичных выступлений. 
        Твоя задача - помогать людям улучшать свои ораторские способности, 
        давать советы по структуре речи, стилю выступления, взаимодействию с аудиторией 
        и другим аспектам ораторского мастерства. 
        Если пользователь предоставляет информацию об анализе своего выступления, 
        используй эти данные для более точных и персонализированных рекомендаций."""
    }
    formatted_messages.append(system_message)
    
    # Добавим сообщения пользователя
    for msg in chat_request.messages:
        formatted_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    print(f"   Форматированных сообщений: {len(formatted_messages)}")
    
    # Вызов аутентификации (это теперь происходит в эндпоинтах)
    print("\n4. Проверка и обновление токена аутентификации...")
    await mock_client.authenticate()
    print("   ✅ Аутентификация выполнена успешно")
    
    # Отправка запроса
    chat_url = f"{mock_client.api_url}/chat/completions"
    request_data = {
        "model": mock_client.model,
        "messages": formatted_messages,
        "temperature": chat_request.temperature,
        "max_tokens": min(chat_request.max_tokens, mock_client.max_tokens),
    }

    headers = {
        "Authorization": f"Bearer {mock_client._access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    print(f"\n5. Отправка запроса на {chat_url}")
    print(f"   Заголовки: Authorization: Bearer {headers['Authorization'][7:17]}...")
    
    # Имитация ответа от API
    response = await mock_client.client.post(chat_url, json=request_data, headers=headers)
    result = await response.json()  # Добавлен await
    
    print("\n6. Обработка ответа от GigaChat...")
    if "choices" in result and len(result["choices"]) > 0:
        assistant_response = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        print(f"   Ответ: {assistant_response}")
        print(f"   Использовано токенов: {tokens_used}")
    else:
        print("   ❌ Не удалось получить ответ от GigaChat")
    
    print("\n" + "="*50)
    print("Демонстрация завершена!")
    print("\nКлючевые особенности реализации:")
    print("- Автоматическая проверка токена перед каждым запросом")
    print("- Обновление токена при необходимости")
    print("- Добавление контекста ораторского искусства")
    print("- Обработка ошибок и валидация данных")


if __name__ == "__main__":
    asyncio.run(demonstrate_chat_functionality())