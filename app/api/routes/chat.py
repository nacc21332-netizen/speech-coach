from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import List, Optional
import logging
from pydantic import BaseModel
import os
from pathlib import Path

from app.services.gigachat import GigaChatClient, GigaChatError
from app.api.deps import get_gigachat_client

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Модель сообщения в чате"""
    role: str  # "user" или "assistant"
    content: str


class ChatRequest(BaseModel):
    """Модель запроса на чат с Gigachat"""
    messages: List[ChatMessage]
    analysis_id: Optional[str] = None  # ID анализа, если пользователь хочет обсудить конкретный анализ
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000


class ChatResponse(BaseModel):
    """Модель ответа от Gigachat"""
    response: str
    model: str
    tokens_used: Optional[int] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_gigachat(
    request: ChatRequest,
    gigachat_client: GigaChatClient = Depends(get_gigachat_client)
):
    """
    Чат с GigaChat - позволяет задавать вопросы и получать ответы от GigaChat
    """
    # Проверяем, что GigaChat клиент доступен
    if gigachat_client is None:
        raise HTTPException(status_code=500, detail="GigaChat service is not available or properly configured")
    
    try:
        # Проверяем, что хотя бы одно сообщение есть
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
        # Подготовим сообщения для отправки в GigaChat
        formatted_messages = []
        
        # Добавим системное сообщение, если это первый запрос
        if request.messages[0].role != "system":
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
        
        # Добавляем сообщения пользователя
        for msg in request.messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Убедимся, что токен аутентификации действителен
        try:
            await gigachat_client.authenticate()
        except GigaChatError as e:
            logger.error(f"Failed to authenticate with GigaChat API: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to authenticate with GigaChat API: {str(e)}")
        
        # Проверяем, что токен действительно установлен после аутентификации
        if not gigachat_client._access_token:
            logger.error("Failed to obtain access token for chat request")
            raise HTTPException(status_code=500, detail="Failed to authenticate with GigaChat API")
        
        # Отправляем запрос в GigaChat
        chat_url = f"{gigachat_client.api_url}/chat/completions"
        
        request_data = {
            "model": gigachat_client.model,
            "messages": formatted_messages,
            "temperature": request.temperature,
            "max_tokens": min(request.max_tokens, gigachat_client.max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {gigachat_client._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        logger.info(f"Sending chat request to GigaChat with {len(formatted_messages)} messages...")
        
        try:
            response = await gigachat_client.client.post(chat_url, json=request_data, headers=headers)
        except httpx.ConnectError as e:
            logger.error(f"Connection error when sending chat request to GigaChat: {e}")
            raise HTTPException(status_code=500, detail=f"Connection error when connecting to GigaChat API: {str(e)}")

        if response.status_code != 200:
            logger.error(f"GigaChat API error {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail=f"GigaChat API error: {response.text}")

        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            logger.error("No choices in GigaChat response")
            raise HTTPException(status_code=500, detail="No response from GigaChat")

        assistant_response = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens")

        logger.info(f"GigaChat responded with {len(assistant_response)} characters")
        
        return ChatResponse(
            response=assistant_response,
            model=gigachat_client.model,
            tokens_used=tokens_used
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_with_gigachat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during chat with GigaChat")


@router.post("/chat/analyze-followup", response_model=ChatResponse)
async def analyze_followup_chat(
    request: ChatRequest,
    gigachat_client: GigaChatClient = Depends(get_gigachat_client)
):
    """
    Чат с GigaChat по результатам анализа - позволяет задавать уточняющие вопросы 
    по конкретному анализу речи
    """
    # Проверяем, что GigaChat клиент доступен
    if gigachat_client is None:
        raise HTTPException(status_code=500, detail="GigaChat service is not available or properly configured")
    
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
        # Если указан analysis_id, можем получить контекст анализа
        # (в реальной реализации здесь будет логика получения сохраненного анализа)
        
        # Подготовим сообщения
        formatted_messages = []
        
        # Системное сообщение с контекстом
        system_content = """Ты - опытный тренер по ораторскому искусству. 
        Пользователь хочет обсудить результаты анализа своего выступления. 
        Отвечай на вопросы пользователя, основываясь на его анализе речи и предоставляя 
        конкретные рекомендации по улучшению. Если пользователь спрашивает о чем-то, 
        что не отражено в анализе, используй свой опыт в области ораторского искусства 
        для предоставления полезных советов."""
        
        formatted_messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Добавляем сообщения пользователя
        for msg in request.messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Убедимся, что токен аутентификации действителен
        try:
            await gigachat_client.authenticate()
        except GigaChatError as e:
            logger.error(f"Failed to authenticate with GigaChat API: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to authenticate with GigaChat API: {str(e)}")
        
        # Проверяем, что токен действительно установлен после аутентификации
        if not gigachat_client._access_token:
            logger.error("Failed to obtain access token for analysis follow-up chat request")
            raise HTTPException(status_code=500, detail="Failed to authenticate with GigaChat API")
        
        # Отправляем запрос в GigaChat
        chat_url = f"{gigachat_client.api_url}/chat/completions"
        
        request_data = {
            "model": gigachat_client.model,
            "messages": formatted_messages,
            "temperature": request.temperature,
            "max_tokens": min(request.max_tokens, gigachat_client.max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {gigachat_client._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        logger.info(f"Sending analysis follow-up request to GigaChat...")
        
        try:
            response = await gigachat_client.client.post(chat_url, json=request_data, headers=headers)
        except httpx.ConnectError as e:
            logger.error(f"Connection error when sending analysis follow-up request to GigaChat: {e}")
            raise HTTPException(status_code=500, detail=f"Connection error when connecting to GigaChat API: {str(e)}")

        if response.status_code != 200:
            logger.error(f"GigaChat API error {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail=f"GigaChat API error: {response.text}")

        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            logger.error("No choices in GigaChat response")
            raise HTTPException(status_code=500, detail="No response from GigaChat")

        assistant_response = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens")

        logger.info(f"GigaChat responded with {len(assistant_response)} characters")
        
        return ChatResponse(
            response=assistant_response,
            model=gigachat_client.model,
            tokens_used=tokens_used
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_followup_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during analysis follow-up chat")


@router.get("/chat/ui", response_class=HTMLResponse)
async def get_chat_page():
    """
    Возвращает HTML-страницу для чата с GigaChat
    """
    try:
        # Определяем путь к файлу шаблона
        template_path = Path(__file__).parent.parent / "templates" / "chat.html"
        
        if not template_path.exists():
            # Если файл не найден, возвращаем простую HTML-страницу
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Чат с GigaChat</title>
                <meta charset="utf-8">
            </head>
            <body>
                <h1>Чат с GigaChat</h1>
                <p>Для полноценного использования чата, пожалуйста, используйте API-запросы к /chat или /chat/analyze-followup</p>
            </body>
            </html>
            """
        else:
            # Читаем содержимое файла шаблона
            with open(template_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
        
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        logger.error(f"Error serving chat page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading chat page")