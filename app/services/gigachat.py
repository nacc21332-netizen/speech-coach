import os
import json
import re
import logging
import uuid
import time
from typing import Optional, Dict, Any, List
import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.models.gigachat import GigaChatAnalysis
from app.models.analysis import AnalysisResult

logger = logging.getLogger(__name__)


class GigaChatError(Exception):
    """Кастомная ошибка для GigaChat API"""
    pass


def should_verify_ssl() -> bool:
    """Определяет, нужно ли проверять SSL сертификаты"""
    # Проверяем переменную окружения
    verify_env = os.environ.get('GIGACHAT_VERIFY_SSL', '').lower()

    if verify_env in ['false', '0', 'no']:
        logger.info("SSL verification disabled by environment variable")
        return False
    elif verify_env in ['true', '1', 'yes']:
        logger.info("SSL verification enabled by environment variable")
        return True

    # По умолчанию для тестирования отключаем SSL проверку
    # В продакшене это нужно включить!
    logger.warning("SSL verification disabled by default for testing")
    logger.warning("Set GIGACHAT_VERIFY_SSL=true for production")
    return False


class GigaChatClient:
    """Клиент для работы с GigaChat API"""

    def __init__(self, verify_ssl: Optional[bool] = None):
        self.api_key = settings.gigachat_api_key.get_secret_value(
        ) if settings.gigachat_api_key else None
        self.auth_url = settings.gigachat_auth_url
        self.api_url = settings.gigachat_api_url
        self.model = settings.gigachat_model
        self.timeout = settings.gigachat_timeout
        self.max_tokens = settings.gigachat_max_tokens
        self.scope = settings.gigachat_scope

        if verify_ssl is None:
            self.verify_ssl = should_verify_ssl()
        else:
            self.verify_ssl = verify_ssl

        if not self.verify_ssl:
            logger.warning("SSL verification is DISABLED! This is insecure!")
            import warnings
            warnings.filterwarnings(
                'ignore', message='Unverified HTTPS request')

        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None

        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            verify=self.verify_ssl,
            limits=httpx.Limits(
                max_keepalive_connections=5, max_connections=10)
        )

    async def authenticate(self) -> None:
        """Аутентификация в GigaChat API"""
        if not self.api_key:
            raise GigaChatError("GigaChat API key not configured")

        # Проверяем, не истек ли текущий токен
        if self._access_token and self._token_expires_at:
            if time.time() < self._token_expires_at - 60:  # 60 секунд до истечения
                logger.debug("Using cached access token")
                return

        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self.api_key}"
            }

            data = {"scope": self.scope}

            logger.info(
                f"Authenticating to GigaChat API (SSL: {self.verify_ssl})")

            auth_response = await self.client.post(
                self.auth_url,
                headers=headers,
                data=data
            )

            if auth_response.status_code == 429:
                # Ошибка 429 - слишком много запросов
                logger.warning(
                    "GigaChat rate limit exceeded (429 Too Many Requests)")
                logger.warning("Waiting and retrying...")

                # Ждем 30 секунд и пробуем еще раз
                import asyncio
                await asyncio.sleep(30)

                # Повторная попытка
                auth_response = await self.client.post(
                    self.auth_url,
                    headers=headers,
                    data=data
                )

            if auth_response.status_code != 200:
                logger.error(f"Auth failed with status {
                             auth_response.status_code}: {auth_response.text}")

                # Если все еще ошибка после повторной попытки
                if auth_response.status_code == 429:
                    raise GigaChatError(
                        "GigaChat rate limit exceeded. Please try again later.")
                else:
                    auth_response.raise_for_status()

            auth_result = auth_response.json()
            self._access_token = auth_result.get("access_token")
            self._token_expires_at = auth_result.get("expires_at")

            if not self._access_token:
                logger.error(f"No access_token in response: {auth_result}")
                raise GigaChatError("Failed to obtain access token")

            logger.info("GigaChat authentication successful")

        except httpx.RequestError as e:
            logger.error(f"GigaChat authentication request failed: {e}")

            if isinstance(e, httpx.ConnectError) and "SSL" in str(e) and self.verify_ssl:
                logger.warning("SSL error, retrying without verification...")
                await self._retry_without_ssl(headers, data)
            else:
                raise GigaChatError(f"Authentication request failed: {e}")

        except Exception as e:
            logger.error(f"GigaChat authentication error: {e}")
            raise GigaChatError(f"Authentication error: {e}")

    async def _retry_without_ssl(self, headers: dict, data: dict):
        """Повторяет аутентификацию без проверки SSL"""
        logger.warning("Creating new client with SSL verification disabled")

        await self.client.aclose()
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            verify=False,
            limits=httpx.Limits(
                max_keepalive_connections=5, max_connections=10)
        )

        auth_response = await self.client.post(
            self.auth_url,
            headers=headers,
            data=data
        )
        auth_response.raise_for_status()

        auth_result = auth_response.json()
        self._access_token = auth_result.get("access_token")

        if not self._access_token:
            raise GigaChatError("Failed to obtain access token")

        logger.info(
            "GigaChat authentication successful (SSL verification disabled)")
        self.verify_ssl = False

    async def analyze_speech(self, analysis_result: AnalysisResult) -> Optional[GigaChatAnalysis]:
        """
        Отправляет результаты анализа в GigaChat для получения
        расширенных персонализированных рекомендаций.
        """
        if not settings.gigachat_enabled:
            logger.info("GigaChat analysis is disabled")
            return None

        # Пробуем аутентифицироваться, если нужно
        if not self._access_token:
            try:
                await self.authenticate()
            except GigaChatError as e:
                logger.warning(f"Failed to authenticate with GigaChat: {e}")
                return None

        try:
            prompt = self._create_analysis_prompt(analysis_result)

            chat_url = f"{self.api_url}/chat/completions"
            request_data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """Ты эксперт по публичным выступлениям. 
                        Анализируй речь по предоставленным метрикам. 
                        Верни ответ ТОЛЬКО в формате JSON без дополнительного текста.
                        
                        Формат JSON:
                        {
                          "overall_assessment": "текст",
                          "strengths": ["пункт1", "пункт2"],
                          "areas_for_improvement": ["пункт1", "пункт2"],
                          "detailed_recommendations": ["пункт1", "пункт2"],
                          "key_insights": ["пункт1", "пункт2"],
                          "confidence_score": число от 0 до 1
                        }"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": min(self.max_tokens, 1500),  # Ограничиваем длину
                "response_format": {"type": "json_object"}
            }

            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            logger.info("Sending analysis request to GigaChat...")

            response = await self.client.post(chat_url, json=request_data, headers=headers)

            if response.status_code != 200:
                logger.error(f"GigaChat API error {
                             response.status_code}: {response.text}")
                return None

            result = response.json()

            if "choices" not in result or len(result["choices"]) == 0:
                logger.error("No choices in GigaChat response")
                return None

            content = result["choices"][0]["message"]["content"]
            logger.info(f"GigaChat response received: {
                        len(content)} characters")

            # Очищаем и парсим JSON
            cleaned_content = self._clean_json_response(content)

            try:
                parsed_content = json.loads(cleaned_content)

                # Валидируем обязательные поля
                required_fields = ["overall_assessment", "strengths", "areas_for_improvement",
                                   "detailed_recommendations", "key_insights", "confidence_score"]

                for field in required_fields:
                    if field not in parsed_content:
                        parsed_content[field] = "" if field != "confidence_score" else 0.5

                return GigaChatAnalysis(**parsed_content)

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to parse GigaChat response: {e}")
                logger.debug(f"Raw content: {content[:500]}...")

                # Создаем базовый ответ
                return GigaChatAnalysis(
                    overall_assessment="Анализ выполнен, но формат ответа не соответствует ожиданиям",
                    strengths=["Получены метрики анализа речи"],
                    areas_for_improvement=[
                        "Требуется корректировка формата ответа GigaChat"],
                    detailed_recommendations=[
                        "Используйте базовый анализ для детальных метрик"],
                    key_insights=["GigaChat вернул невалидный JSON формат"],
                    confidence_score=0.2
                )

        except httpx.RequestError as e:
            logger.error(f"GigaChat API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in GigaChat analysis: {e}")
            return None

    def _create_analysis_prompt(self, analysis_result: AnalysisResult) -> str:
        """Создает промпт для анализа на основе результатов"""
        filler_items = ""
        for item in analysis_result.filler_words.items:
            if item.get("count", 0) > 0:
                filler_items += f"- {item['word']}: {item['count']} раз\n"

        pauses_info = ""
        if analysis_result.pauses.long_pauses:
            pauses_info = "Длинные паузы:\n"
            for pause in analysis_result.pauses.long_pauses[:3]:
                pauses_info += f"- {pause['duration']:.1f} сек (с {pause['start']:.1f} по {
                    pause['end']:.1f})\n"

        advice_info = ""
        for advice in analysis_result.advice:
            advice_info += f"- {advice.title}: {advice.observation}\n"

        prompt = f"""Проанализируй это публичное выступление:

=== ТРАНСКРИПТ ===
{analysis_result.transcript[:3000]}{'... [текст сокращен]' if len(analysis_result.transcript) > 3000 else ''}

=== МЕТРИКИ ===
Длительность: {analysis_result.duration_sec:.1f} секунд
Время говорения: {analysis_result.speaking_time_sec:.1f} секунд
Коэффициент говорения: {analysis_result.speaking_ratio:.2%}
Темп речи: {analysis_result.words_per_minute:.1f} слов/минуту
Общее количество слов: {analysis_result.words_total}

Слова-паразиты: {analysis_result.filler_words.total} ({analysis_result.filler_words.per_100_words:.1f} на 100 слов)
{f'Наиболее частые:\n{filler_items}' if filler_items else ''}

Количество пауз: {analysis_result.pauses.count}
Средняя длина паузы: {analysis_result.pauses.avg_sec:.1f} секунд
Самая длинная пауза: {analysis_result.pauses.max_sec:.1f} секунд
{pauses_info if pauses_info else ''}

Количество фраз: {analysis_result.phrases.count}
Средняя длина фразы: {analysis_result.phrases.avg_words:.1f} слов
Классификация длины фраз: {analysis_result.phrases.length_classification}
Вариативность ритма: {analysis_result.phrases.rhythm_variation}

=== СТАНДАРТНЫЕ РЕКОМЕНДАЦИИ ===
{advice_info}

Дай развернутый анализ с учетом контекста публичного выступления.
Обрати внимание на:
1. Ясность и структурированность мысли
2. Эмоциональную окраску речи
3. Убедительность аргументации
4. Взаимодействие с аудиторией (на основе пауз и темпа)
5. Профессиональную лексику и терминологию
6. Общее впечатление от выступления

Верни ответ строго в формате JSON, как указано в system prompt."""

        return prompt

    async def close(self):
        """Закрывает HTTP-клиент"""
        try:
            await self.client.aclose()
            logger.debug("GigaChat HTTP клиент закрыт")
        except Exception as e:
            logger.debug(f"Ошибка закрытия HTTP клиента: {e}")

    async def analyze_speech_with_timings(self, timed_result_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Отправляет результаты анализа с таймингами в GigaChat.
        """
        if not settings.gigachat_enabled:
            logger.info("GigaChat detailed analysis is disabled")
            return None

        # Пробуем аутентифицироваться, если нужно
        if not self._access_token:
            try:
                await self.authenticate()
            except GigaChatError as e:
                logger.warning(f"Failed to authenticate with GigaChat: {e}")
                return None

        start_time = time.time()

        try:
            prompt = self._create_detailed_analysis_prompt(timed_result_dict)

            chat_url = f"{self.api_url}/chat/completions"
            request_data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """Ты эксперт по публичным выступлениям, ораторскому искусству и анализу речи. 
                        Твоя задача - анализировать выступление с учетом ТОЧНЫХ ВРЕМЕННЫХ МЕТОК.
                        
                        Требования к анализу:
                        1. Привязывай все рекомендации к конкретным секундам выступления
                        2. Выявляй временные паттерны и закономерности
                        3. Определяй критические моменты (поворотные точки, кульминации)
                        4. Анализируй стиль речи и его уместность
                        5. Оценивай предполагаемую вовлеченность аудитории по времени
                        6. Составь временную шкалу улучшений с упражнениями
                        
                        Отвечай ТОЛЬКО в указанном JSON формате."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": self.max_tokens * 2,
                "response_format": {"type": "json_object"}
            }

            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            logger.info("Sending detailed analysis request to GigaChat...")

            response = await self.client.post(chat_url, json=request_data, headers=headers)

            if response.status_code != 200:
                logger.error(f"GigaChat API error {
                             response.status_code}: {response.text}")
                processing_time = time.time() - start_time
                return self._create_error_response("GigaChat API error", processing_time)

            result = response.json()

            if "choices" not in result or len(result["choices"]) == 0:
                logger.error("No choices in GigaChat response")
                processing_time = time.time() - start_time
                return self._create_error_response("No choices in response", processing_time)

            content = result["choices"][0]["message"]["content"]
            processing_time = time.time() - start_time

            logger.info(f"GigaChat detailed analysis received in {
                        processing_time:.1f} seconds")

            try:
                # Попробуем более терпимый парсинг: сначала очистить текст,
                # затем попытаться распарсить JSON. Это позволит обработать
                # ситуации с лишними запятыми, поясняющими фразами вокруг JSON и т.п.
                cleaned = self._clean_json_response(content)
                parsed_content = json.loads(cleaned)
                parsed_content["processing_time_sec"] = processing_time
                return parsed_content

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to parse GigaChat response as JSON: {e}")
                logger.debug(f"Raw content (first 2000 chars): {content[:2000]}")
                processing_time = time.time() - start_time
                return self._create_error_response(f"JSON parse error", processing_time)

        except httpx.RequestError as e:
            logger.error(f"GigaChat API request failed: {e}")
            processing_time = time.time() - start_time
            return self._create_error_response(f"Request error: {str(e)}", processing_time)
        except Exception as e:
            logger.error(f"Error processing GigaChat response: {e}")
            processing_time = time.time() - start_time
            return self._create_error_response(f"Processing error: {str(e)}", processing_time)

    def _create_detailed_analysis_prompt(self, timed_result: Dict[str, Any]) -> str:
        """Создает детализированный промпт для GigaChat с учетом таймингов"""
        # Извлекаем данные из словаря
        duration_sec = timed_result.get("duration_sec", 0)
        speaking_time_sec = timed_result.get("speaking_time_sec", 0)
        speaking_ratio = timed_result.get("speaking_ratio", 0)
        words_total = timed_result.get("words_total", 0)
        words_per_minute = timed_result.get("words_per_minute", 0)
        transcript = timed_result.get("transcript", "")

        # Извлекаем timeline
        timeline = timed_result.get("timeline", {})
        fillers = timeline.get("fillers", [])
        pauses = timeline.get("pauses", [])
        phrases = timeline.get("phrases", [])
        suspicious_moments = timeline.get("suspicious_moments", [])

        # Базовые метрики
        filler_count = len(fillers)
        pause_count = len(pauses)
        phrase_count = len(phrases)
        problem_count = len(suspicious_moments)

        # Рассчитываем статистику
        filler_per_100 = (filler_count / words_total *
                          100) if words_total > 0 else 0
        problematic_pauses = sum(
            1 for p in pauses if p.get("is_excessive", False))

        prompt = f"""
Ты эксперт по публичным выступлениям и ораторскому искусству.
Анализируй речь по предоставленным метрикам, транскрипту и ДЕТАЛИЗИРОВАННЫМ ТАЙМИНГАМ.

=== ОСНОВНЫЕ МЕТРИКИ ===
Длительность: {duration_sec:.1f} секунд
Время говорения: {speaking_time_sec:.1f} секунд
Коэффициент говорения: {speaking_ratio:.2%}
Темп речи: {words_per_minute:.1f} слов/минуту
Общее количество слов: {words_total}
Слов-паразитов: {filler_count} ({filler_per_100:.1f} на 100 слов)
Пауз: {pause_count} (проблемных: {problematic_pauses})
Фраз: {phrase_count}
Проблемных моментов: {problem_count}

=== ТРАНСКРИПТ (первые 2500 символов) ===
{transcript[:2500]}{'... [текст сокращен]' if len(transcript) > 2500 else ''}

=== ИНСТРУКЦИИ ДЛЯ АНАЛИЗА ===
1. Проанализируй выступление с привязкой ко времени
2. Определи критические моменты (кульминация, поворотные точки)
3. Выяви временные паттерны (когда возникают проблемы, когда речь наиболее эффективна)
4. Оцени стиль речи и его уместность
5. Предположи реакцию аудитории в разные моменты
6. Составь план улучшений с временной привязкой

=== ТРЕБОВАНИЯ К ФОРМАТУ ОТВЕТА ===
Ответ должен быть в формате JSON со следующей структурой:
{{
  "overall_assessment": "Общая оценка (2-3 абзаца)",
  "strengths": ["сильная сторона 1", "сильная сторона 2", ...],
  "areas_for_improvement": ["зона роста 1", "зона роста 2", ...],
  "detailed_recommendations": ["рекомендация 1", "рекомендация 2", ...],
  "key_insights": ["инсайт 1", "инсайт 2", ...],
  "confidence_score": 0.85,
  "time_based_analysis": [],
  "temporal_patterns": [],
  "improvement_timeline": [],
  "critical_moments": []
}}

ВАЖНО:
1. Все временные метки указывай в секундах
2. Будь максимально конкретен в рекомендациях
3. Привязывай все советы к конкретным моментам времени
"""
        return prompt

    def _clean_json_response(self, content: str) -> str:
        """Попытка аккуратно извлечь/очистить JSON из произвольного текста.

        Стратегия:
        - Найти первую '{' и последнюю '}' и взять подстроку
        - Заменить «умные» кавычки на обычные
        - Удалить хвостовые запятые перед '}' и ']' с помощью regex
        - Убрать управляющие символы
        Если ничего не найдено — вернуть исходную строку.
        """
        try:
            if not content or not isinstance(content, str):
                return content

            # Нормализация кавычек
            s = content.replace('“', '"').replace('”', '"').replace("\u2018", "'").replace("\u2019", "'")

            # Найдём самую левую фигурную скобку и самую правую
            first = s.find('{')
            last = s.rfind('}')
            if first != -1 and last != -1 and last > first:
                s = s[first:last+1]

            # Уберём возможные односторонние комменты и управляющие символы
            s = re.sub(r'\s+//.*', '', s)
            s = re.sub(r'\s+\\n', ' ', s)

            # Удаляем хвостовые запятые перед закрывающими скобками
            s = re.sub(r',\s*(?=[}\]])', '', s)

            # Trim
            s = s.strip()

            return s
        except Exception:
            return content

    def _create_error_response(self, error_message: str, processing_time: float) -> Dict[str, Any]:
        """Создает ответ об ошибке"""
        return {
            "overall_assessment": f"Анализ выполнен с ограничениями: {error_message}",
            "strengths": ["Получены базовые метрики анализа"],
            "areas_for_improvement": ["Требуется исправление в работе GigaChat API"],
            "detailed_recommendations": ["Попробуйте использовать базовый анализ или проверьте настройки API"],
            "key_insights": [f"Ошибка: {error_message}"],
            "confidence_score": 0.1,
            "processing_time_sec": processing_time,
            "metadata": {
                "has_error": True,
                "error_message": error_message
            }
        }
