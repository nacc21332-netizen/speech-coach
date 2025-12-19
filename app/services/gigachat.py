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
from app.services.cache import AnalysisCache

logger = logging.getLogger(__name__)


class GigaChatError(Exception):
    """GigaChat API error."""
    pass


def should_verify_ssl() -> bool:
    """Determine whether to verify SSL certificates."""
    verify_env = os.environ.get('GIGACHAT_VERIFY_SSL', '').lower()

    if verify_env in ['false', '0', 'no']:
        logger.warning("SSL verification disabled (not recommended for production)")
        return False
    elif verify_env in ['true', '1', 'yes']:
        logger.info("SSL verification enabled")
        return True

    # Default to enabled for security
    logger.info("SSL verification enabled (default)")
    return True


class GigaChatClient:
    """GigaChat API client."""

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
            logger.warning("⚠️  SSL verification is DISABLED (NOT RECOMMENDED for production)")
            logger.warning("Set GIGACHAT_VERIFY_SSL=true in production")

        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None

        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            verify=self.verify_ssl,
            limits=httpx.Limits(
                max_keepalive_connections=5, max_connections=10)
        )

    async def authenticate(self) -> None:
        """Authenticate to GigaChat API."""
        if not self.api_key:
            raise GigaChatError("GigaChat API key not configured")

        # Check if cached token is still valid
        if self._access_token and self._token_expires_at:
            if time.time() < self._token_expires_at - 60:  # 60 seconds buffer
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

            logger.info(f"Authenticating to GigaChat API")

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
                logger.error(f"Auth failed with status {auth_response.status_code}: {auth_response.text}")

                # Если все еще ошибка после повторной попытки
                if auth_response.status_code == 429:
                    raise GigaChatError(
                        "GigaChat rate limit exceeded. Please try again later.")
                else:
                    auth_response.raise_for_status()

            auth_result = auth_response.json()
            self._access_token = auth_result.get("access_token")
            
            # Получаем время жизни токена из ответа
            expires_in = auth_result.get("expires_in")  # обычно в секундах
            if expires_in:
                # Преобразуем в абсолютное время
                self._token_expires_at = time.time() + expires_in
            else:
                # Если нет expires_in, используем expires_at (если это абсолютное время)
                expires_at = auth_result.get("expires_at")
                if isinstance(expires_at, (int, float)):
                    # Проверяем, является ли это timestamp (большое число) или временем жизни (меньше 1 дня)
                    if expires_at > time.time() + 86400:  # больше чем через день
                        # Это абсолютное время
                        self._token_expires_at = expires_at
                    else:
                        # Это время жизни, преобразуем в абсолютное
                        self._token_expires_at = time.time() + expires_at
                else:
                    # По умолчанию устанавливаем время истечения через 9 минут (540 секунд)
                    self._token_expires_at = time.time() + 540

            if not self._access_token:
                logger.error(f"No access_token in response: {auth_result}")
                raise GigaChatError("Failed to obtain access token")

            logger.info("GigaChat authentication successful")

        except httpx.ConnectError as e:
            logger.error(f"GigaChat connection error: {e}")
            logger.error(f"Could not connect to GigaChat API at {self.auth_url}")
            logger.error("Please check your network connection and API endpoint configuration")
            raise GigaChatError(f"Connection failed to GigaChat API: {e}")
        
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
        
        # Устанавливаем время истечения токена также и в этом случае
        expires_in = auth_result.get("expires_in")  # обычно в секундах
        if expires_in:
            # Преобразуем в абсолютное время
            self._token_expires_at = time.time() + expires_in
        else:
            # Если нет expires_in, используем expires_at (если это абсолютное время)
            expires_at = auth_result.get("expires_at")
            if isinstance(expires_at, (int, float)):
                # Проверяем, является ли это timestamp (большое число) или временем жизни (меньше 1 дня)
                if expires_at > time.time() + 86400:  # больше чем через день
                    # Это абсолютное время
                    self._token_expires_at = expires_at
                else:
                    # Это время жизни, преобразуем в абсолютное
                    self._token_expires_at = time.time() + expires_at
            else:
                # По умолчанию устанавливаем время истечения через 9 минут (540 секунд)
                self._token_expires_at = time.time() + 540

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

            # Убедимся, что токен аутентификации действителен
            await self.authenticate()
            
            # Проверяем, что токен действительно установлен после аутентификации
            if not self._access_token:
                logger.error("Failed to obtain access token for analysis request")
                return None

            chat_url = f"{self.api_url}/chat/completions"

            request_data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """Ты инструктор по публичным выступлениям, обучающий студентов навыкам ораторского мастерства.

Входные данные: транскрипт выступления + объективные метрики (темп, паузы, паразиты).

Твоя задача: 
1) Анализируй транскрипт как целое (структура, логика, ясность)
2) Используй метрики как подтверждение/опровержение твоих выводов
3) Дай КОНКРЕТНЫЕ, ВЫПОЛНИМЫЕ рекомендации

Критерии оценки (в порядке важности):
- СОДЕРЖАНИЕ (организация, логика, полнота): 40%
- ДОСТАВКА (беглость, уверенность, плавность): 30%
- ЯСНОСТЬ (доступность языка, отсутствие двусмысленностей): 20%
- ВЛИЯНИЕ (запоминаемость, убедительность): 10%

Выходной формат: JSON. Структура точно как указано ниже.

ЗАПРЕТЫ:
- ❌ Не придумывай примеры, которых нет в транскрипте
- ❌ Не добавляй метрики, которых я не дал
- ❌ Если данных недостаточно, скажи "недостаточно информации" вместо выдумки
- ❌ Не добавляй текст ДО JSON, не добавляй текст ПОСЛЕ JSON
- ❌ Используй ТОЛЬКО двойные кавычки для строк в JSON

КРИТИЧЕСКИЕ ПРАВИЛА:
- ✅ Ссылайся на конкретные фрагменты транскрипта
- ✅ Используй цифры из метрик, не придумывай новые
- ✅ Каждая рекомендация должна быть ДЕЙСТВЕННОЙ (можно ли ее выполнить за неделю?)
- ✅ Баланс похвалы и критики: минимум 50% хороших замечаний
- ✅ Твой ответ ДОЛЖЕН быть валидным JSON без исключений"""
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                # Use configured max tokens (no artificial 2000 cap)
                "max_tokens": int(self.max_tokens),
                "response_format": {"type": "json_object"},
            }

            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            logger.info("Sending analysis request to GigaChat...")

            response = await self.client.post(chat_url, json=request_data, headers=headers)

            if response.status_code != 200:
                logger.error(f"GigaChat API error {response.status_code}: {response.text}")
                return None

            result = response.json()

            if "choices" not in result or len(result["choices"]) == 0:
                logger.error("No choices in GigaChat response")
                return None

            content = result["choices"][0]["message"]["content"]
            logger.info(f"GigaChat response received: {len(content)} characters")

            # Пробуем распарсить JSON с несколькими попытками
            parsed_content = self._parse_json_with_retries(content)
            
            if parsed_content is not None:
                # Валидируем обязательные поля (новая структура)
                validated = self._validate_and_normalize_analysis(parsed_content)
                
                if validated is not None:
                    return GigaChatAnalysis(**validated)
                else:
                    logger.warning("JSON parsed but validation failed")
                    return self._create_fallback_analysis("JSON валидация не прошла")
            else:
                logger.warning("Failed to parse GigaChat response after retries")
                logger.debug(f"Raw content: {content[:500]}...")
                return self._create_fallback_analysis("Невалидный формат JSON от GigaChat")

        except httpx.RequestError as e:
            logger.error(f"GigaChat API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in GigaChat analysis: {e}")
            return None

    def _create_analysis_prompt(self, analysis_result: AnalysisResult) -> str:
        """Создает оптимизированный промпт для анализа (вариант 3: Recommended)"""
        filler_items = ""
        for item in analysis_result.filler_words.items:
            if item.get("count", 0) > 0:
                filler_items += f"- {item['word']}: {item['count']} раз\n"

        pauses_info = ""
        if analysis_result.pauses.long_pauses:
            pauses_info = ""
            for pause in analysis_result.pauses.long_pauses[:3]:
                pauses_info += f"- {pause['duration']:.1f} сек (с {pause['start']:.1f} по {pause['end']:.1f})\n"

        # Интерпретация темпа
        wpm = analysis_result.words_per_minute
        tempo_interpretation = "очень быстро (> 160)" if wpm > 160 else "быстро (140-160)" if wpm > 140 else "оптимально (120-140)" if wpm > 120 else "медленно (< 120)"

        # Интерпретация длины фраз
        avg_phrase_len = analysis_result.phrases.avg_words
        phrase_interpretation = "короткие фразы (< 5 слов) - может быть отрывисто" if avg_phrase_len < 5 else "средние (5-10) - норма" if avg_phrase_len < 10 else "длинные (> 10) - возможна усталость слушателя"

        # Интерпретация паразитов
        fillers_per_100 = analysis_result.filler_words.per_100_words
        filler_interpretation = "ОТЛИЧНО" if fillers_per_100 < 2 else "ХОРОШО" if fillers_per_100 < 3 else "ТРЕБУЕТ РАБОТЫ" if fillers_per_100 < 5 else "КРИТИЧНО"

        # Интерпретация пауз
        max_pause = analysis_result.pauses.max_sec
        pause_interpretation = "очень длинные паузы - может выглядеть как растерянность" if max_pause > 3 else "нормальные паузы - хорошо" if max_pause > 1 else "очень короткие - мало дышит"

        prompt = f"""АНАЛИЗИРУЙ ВЫСТУПЛЕНИЕ СТУДЕНТА:

═══════════════════════════════════════════════════════════════════════════════
ОБЪЕКТИВНЫЕ ДАННЫЕ
═══════════════════════════════════════════════════════════════════════════════
Длительность выступления: {analysis_result.duration_sec:.1f} секунд
Всего слов в выступлении: {analysis_result.words_total}

ТЕМП РЕЧИ И БЕГЛОСТЬ:
• Скорость речи: {wpm:.1f} слов/минуту
  (Интерпретация: {tempo_interpretation})

• Средняя длина фразы: {avg_phrase_len:.1f} слов
  (Интерпретация: {phrase_interpretation})

• Разнообразие темпа: {analysis_result.phrases.rhythm_variation}
  
ДИСФЛЮЕНТНОСТЬ (признаки неуверенности):
• Слова-паразиты: {analysis_result.filler_words.total} всего
• Частота паразитов: {fillers_per_100:.1f} на 100 слов
  (Норма: < 2 на 100 слов | Текущий уровень: {filler_interpretation})

{f'• Самые частые: {filler_items}' if filler_items else ''}

ПАУЗЫ (стратегия пауз):
• Количество пауз: {analysis_result.pauses.count}
• Средняя длина: {analysis_result.pauses.avg_sec:.2f} сек
• Максимальная: {max_pause:.2f} сек
  (Интерпретация: {pause_interpretation})

{f'• {pauses_info}' if pauses_info else ''}

═══════════════════════════════════════════════════════════════════════════════
ПОЛНЫЙ ТРАНСКРИПТ (для анализа содержания)
═══════════════════════════════════════════════════════════════════════════════
{analysis_result.transcript}

═══════════════════════════════════════════════════════════════════════════════
ТВОЙ АНАЛИЗ: Ответь на эти вопросы мысленно перед JSON'ом
═══════════════════════════════════════════════════════════════════════════════

1️⃣ СТРУКТУРА (читаешь весь транскрипт снизу вверху):
   - Четкое ли начало? Что первое слово/фраза?
   - Развиваются ли идеи? Есть ли "поворотные моменты"?
   - Сильное ли завершение? Или просто обрывается?
   - Есть ли скрытые переходы ("во-первых", "следовательно", "в итоге")?

2️⃣ СОДЕРЖАНИЕ (выделяешь основные идеи):
   - Главная идея (в одном предложении)?
   - Поддерживающие идеи? Примеры? Доказательства?
   - Есть ли пустые места (идея высказана, но не развита)?

3️⃣ СОГЛАСОВАННОСТЬ МЕТРИК И СОДЕРЖАНИЯ:
   - Быстрый темп ({wpm:.0f} слов/мин) указывает на спешку или волнение?
   - Паразиты ({fillers_per_100:.1f}/100) говорят о нерешительности?
   - Пауз ({analysis_result.pauses.count} шт) — это обдумывание или неуверенность?

4️⃣ ЯЗЫК И ДОСТУПНОСТЬ:
   - Сложные ли термины? Объясняются ли?
   - Предложения понятные? Не слишком сложные ли?
   - Есть ли коллоквиализмы/жаргон?

5️⃣ ВЕРНИ СТРУКТУРИРОВАННЫЙ JSON (строго как ниже, БЕЗ КАКИХ-ЛИБО ТЕКСТОВ ПЕРЕД ИЛИ ПОСЛЕ):

{{
    "выступление_анализ": {{
        "общее_впечатление": "1-2 предложения о выступлении в целом",
        "главная_идея": "Опиши в одном предложении основной месседж",
        "оценка_из_100": число от 1 до 100
    }},
    
    "структура_и_организация": {{
        "есть_ли_четкая_структура": "да/нет + объяснение (2-3 предложения)",
        "введение": "как оратор начинает? эффективное ли?",
        "основная_часть": "развиваются ли идеи логично? найди 1-2 примера из текста",
        "заключение": "сильное ли завершение? запоминается ли?",
        "оценка": число 1-10
    }},
    
    "содержание": {{
        "основные_идеи": ["идея 1", "идея 2", "идея 3"],
        "примеры_и_доказательства": "есть ли подтверждение каждой идеи? достаточно?",
        "пропуски_или_слабости": ["слабость 1", "слабость 2"] или [],
        "оценка": число 1-10
    }},
    
    "язык_и_доступность": {{
        "уровень_сложности": "простой/средний/сложный",
        "проблемные_места": ["термин который не объяснен", "слишком сложное предложение"] или [],
        "оценка": число 1-10
    }},
    
    "доставка_и_беглость": {{
        "интерпретация_темпа": "{wpm:.1f} слов/мин означает {tempo_interpretation}",
        "интерпретация_паразитов": "{fillers_per_100:.1f}/100 слов указывает на {filler_interpretation}",
        "интерпретация_пауз": "паузы используются так. Это [хорошо/плохо] потому что [объяснение]",
        "общая_оценка_доставки": "уверенный/нервный/размеренный голос, [почему?]",
        "оценка": число 1-10
    }},
    
    "сильные_стороны": [
        "Сильная сторона 1 (с примером из текста выступления)",
        "Сильная сторона 2",
        "Сильная сторона 3"
    ],
    
    "области_для_улучшения": [
        {{
            "проблема": "ЧТО нужно улучшить (четко и конкретно)",
            "причина": "ПОЧЕМУ это проблема",
            "решение": "КАК это исправить (действенный совет)"
        }},
        {{
            "проблема": "...",
            "причина": "...",
            "решение": "..."
        }}
    ],
    
    "главные_рекомендации": [
        "Рекомендация 1: [действенный совет, который можно выполнить за неделю]",
        "Рекомендация 2: [...]",
        "Рекомендация 3: [...]",
        "Рекомендация 4: [...]"
    ],
    
    "приоритет_развития": "Какой ОДИН навык улучшить в первую очередь? (с объяснением)",
    
    "уровень_уверенности": число от 0 до 1
}}

ВАЖНО: 
- Каждое утверждение подкреплено примерами из транскрипта
- Рекомендации действенные, не общие фразы
- Баланс: 50% похвалы, 50% критики
- Используй метрики как подтверждение выводов"""

        return prompt

    async def close(self):
        """Закрывает HTTP-клиент"""
        try:
            await self.client.aclose()
            logger.debug("GigaChat HTTP клиент закрыт")
        except Exception as e:
            logger.debug(f"Ошибка закрытия HTTP клиента: {e}")

    async def classify_fillers_context(self, contexts: List[Dict[str, Any]], cache: Optional[AnalysisCache] = None) -> List[Dict[str, Any]]:
        """Classify each candidate filler word in context using LLM with caching and retry/backoff.
        Returns enriched list of contexts with `is_filler_context` and `score`.
        """
        if not settings.llm_fillers_enabled:
            return [dict(**c, is_filler=False, confidence=0.0, reason="llm_disabled", suggestion=None) for c in contexts]

        if not self._access_token:
            try:
                await self.authenticate()
            except GigaChatError as e:
                logger.warning(f"Failed to authenticate for filler classification: {e}")
                return [dict(**c, is_filler=False, confidence=0.0) for c in contexts]

        import hashlib
        import json as _json
        import asyncio as _asyncio

        def _ctx_key(c):
            s = _json.dumps({
                "word": c.get("word"),
                "exact_word": c.get("exact_word"),
                "context_before": c.get("context_before", ""),
                "context_after": c.get("context_after", "")
            }, ensure_ascii=False, sort_keys=True)
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        results: List[Dict[str, Any]] = []
        to_query = []
        idx_map = {}
        for i, c in enumerate(contexts):
            key = _ctx_key(c)
            cached = None
            if cache is not None:
                try:
                    cached = cache.get_by_key(key)
                except Exception:
                    cached = None

            if cached is not None:
                results.append(dict(**c, is_filler=cached.get("is_filler", False), confidence=cached.get("confidence", 0.0)))
            else:
                idx_map[len(to_query)] = i
                to_query.append((key, c))

        if not to_query:
            return results

        # Build prompt and request for to_query
        items_block = _json.dumps([{
            "index": i + 1,
            "word": q[1].get('word'),
            "exact_word": q[1].get('exact_word'),
            "context_before": q[1].get('context_before', ''),
            "context_after": q[1].get('context_after', ''),
            "timestamp": q[1].get('timestamp', 0.0)
        } for i, q in enumerate(to_query)], ensure_ascii=False)

        # More structured, conservative instruction for contextual filler classification.
        user_prompt = (
            "Ты эксперт по анализу устной речи. Для каждого элемента из списка оцени, "
            "является ли указанное слово в данном контексте словом-паразитом (лишним дисфункциональным маркером), "
            "или выполняет дискурсивную/семантическую функцию (подтверждение, указание места, ответ и т.п.). "
            "Будь консервативен — отмечай как слово-паразит только когда слово действительно не несет смысловой нагрузки и его удаление/замена улучшит поток речи. "
            "Для каждой позиции верни JSON объект с полями: \n"
            "- index: исходный индекс (целое число)\n"
            "- is_filler: true или false\n"
            "- confidence: число от 0 до 1 (чем выше, тем увереннее)\n"
            "- reason: короткая фраза (1-2 предложения), почему это слово является или не является паразитом\n"
            "- suggestion: если is_filler=true — предложи конкретную замену (короткая пауза 0.2-0.6с, связующая фраза, или конкретное слово).\n\n"
            f"Обработай эти элементы: {items_block}\n"
            "Верни строго корректный JSON-массив без лишних комментариев."
        )

        chat_url = f"{self.api_url}/chat/completions"
        request_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Ты ассистент, помогающий классифицировать слова-паразиты в контексте выступления. Отвечай в JSON-формате только ответом пользователя."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": min(self.max_tokens, settings.llm_fillers_max_tokens),
            "response_format": {"type": "json_array"}
        }

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Retry loop
        max_retries = 3
        backoff = 1
        response = None
        for attempt in range(max_retries):
            try:
                response = await self.client.post(chat_url, json=request_data, headers=headers)
                if response.status_code == 200:
                    break
                elif response.status_code in (429, 503):
                    logger.warning(f"GigaChat rate-limited (status {response.status_code}), retrying in {backoff}s")
                    await _asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logger.error(f"GigaChat classify API error {response.status_code}: {response.text}")
                    return [dict(**c, is_filler=False, confidence=0.0) for c in contexts]
            except httpx.ConnectError as e:
                logger.warning(f"GigaChat classify connection error (attempt {attempt+1}): {e}")
                logger.warning(f"Could not connect to GigaChat API at {chat_url}")
                await _asyncio.sleep(backoff)
                backoff *= 2
                continue
            except Exception as e:
                logger.warning(f"GigaChat classify request failed (attempt {attempt+1}): {e}")
                await _asyncio.sleep(backoff)
                backoff *= 2
                continue

        if response is None:
            logger.error("GigaChat classify API failed after retries")
            return [dict(**c, is_filler=False, confidence=0.0) for c in contexts]

        try:
            body = response.json()
            if not body.get("choices"):
                return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]

            # Extract content from response
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]

            # Parse JSON response (handling potential malformed JSON)
            parsed_content = self._parse_json_with_retries(content)
            if parsed_content is None:
                logger.warning("Failed to parse LLM response for filler classification")
                return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]

            # Ensure it's a list
            if not isinstance(parsed_content, list):
                parsed_content = [parsed_content]

            # Build result by mapping LLM responses back to original contexts
            # Build a map of query index to original index
            llm_results_by_index = {}
            for llm_result in parsed_content:
                idx = llm_result.get("index", 0)
                if idx > 0 and idx <= len(to_query):
                    llm_results_by_index[idx - 1] = llm_result

            # Populate results for queried items and cache them
            for query_idx, (key, c) in enumerate(to_query):
                llm_result = llm_results_by_index.get(query_idx, {})
                is_filler = llm_result.get("is_filler", False)
                confidence = llm_result.get("confidence", 0.0)
                reason = llm_result.get("reason")
                suggestion = llm_result.get("suggestion")

                result_dict = dict(**c,
                                   is_filler=bool(is_filler),
                                   confidence=float(confidence),
                                   reason=reason,
                                   suggestion=suggestion)
                results.append(result_dict)

                # Cache the result
                if cache is not None:
                    try:
                        cache.put_by_key(key, {
                            "is_filler": is_filler,
                            "confidence": confidence,
                            "reason": reason,
                            "suggestion": suggestion
                        })
                    except Exception as e:
                        logger.debug(f"Failed to cache filler classification: {e}")

            return results

        except Exception as e:
            logger.warning(f"Failed to process filler classification response: {e}")
            return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]

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
            s = re.sub(r'\\n', ' ', s)

            # Удаляем хвостовые запятые перед закрывающими скобками
            s = re.sub(r',\s*(?=[}\]])', '', s)

            # Trim
            s = s.strip()

            return s
        except Exception:
            return content

    def _validate_and_normalize_analysis(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Валидирует и нормализирует ответ анализа для новой структуры (вариант 3).
        
        Поддерживает оба формата:
        - Старый: overall_assessment, strengths, areas_for_improvement, detailed_recommendations, key_insights
        - Новый: выступление_анализ, структура_и_организация, содержание, язык_и_доступность, доставка_и_беглость
        
        Returns:
            Нормализированный словарь для GigaChatAnalysis или None если невалидно
        """
        try:
            # Определяем формат ответа
            is_new_format = "выступление_анализ" in data
            
            if is_new_format:
                # Конвертируем новый формат в старый для совместимости
                analyzed = data.get("выступление_анализ", {})
                
                overall = analyzed.get("общее_впечатление", "")
                strengths = data.get("сильные_стороны", [])
                improvements = []
                
                # Преобразуем области улучшения из новой структуры
                areas = data.get("области_для_улучшения", [])
                if isinstance(areas, list):
                    for area in areas:
                        if isinstance(area, dict):
                            improvements.append(f"{area.get('проблема', '')} → {area.get('решение', '')}")
                        else:
                            improvements.append(str(area))
                
                recommendations = data.get("главные_рекомендации", [])
                insights = [data.get("приоритет_развития", "")]
                confidence = data.get("уровень_уверенности", 0.7)
                
                normalized = {
                    "overall_assessment": overall if overall else "Анализ выступления",
                    "strengths": strengths if strengths else ["Выполнен анализ речи"],
                    "areas_for_improvement": improvements if improvements else ["Требуется улучшение"],
                    "detailed_recommendations": recommendations if recommendations else ["Работайте над рекомендациями"],
                    "key_insights": [ins for ins in insights if ins],
                    "confidence_score": float(confidence) if confidence else 0.7
                }
            else:
                # Старый формат - просто валидируем
                normalized = {
                    "overall_assessment": str(data.get("overall_assessment", ""))[:1000],
                    "strengths": data.get("strengths", []) or ["Анализ выполнен"],
                    "areas_for_improvement": data.get("areas_for_improvement", []) or [],
                    "detailed_recommendations": data.get("detailed_recommendations", []) or [],
                    "key_insights": data.get("key_insights", []) or [],
                    "confidence_score": float(data.get("confidence_score", 0.7))
                }
            
            # Финальная валидация полей
            if not normalized["overall_assessment"]:
                normalized["overall_assessment"] = "Анализ публичного выступления выполнен"
            
            if not normalized["strengths"]:
                normalized["strengths"] = ["Выполнен полный анализ речи"]
            
            if not isinstance(normalized["confidence_score"], (int, float)):
                normalized["confidence_score"] = 0.7
            
            # Убедимся что значение confidence_score в диапазоне 0-1
            normalized["confidence_score"] = max(0, min(1, float(normalized["confidence_score"])))
            
            return normalized
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return None

    def _create_fallback_analysis(self, error_reason: str) -> Optional[GigaChatAnalysis]:
        """Создает fallback анализ при ошибке"""
        try:
            return GigaChatAnalysis(
                overall_assessment=f"Анализ выполнен с ограничениями: {error_reason}",
                strengths=["Получены объективные метрики анализа речи"],
                areas_for_improvement=["Требуется улучшение работы с GigaChat API"],
                detailed_recommendations=["Проверьте настройки API", "Используйте базовый анализ"],
                key_insights=["Используйте доступные метрики для анализа"],
                confidence_score=0.3
            )
        except Exception as e:
            logger.error(f"Failed to create fallback analysis: {e}")
            return None

    def _parse_json_with_retries(self, content: str, max_retries: int = 3):
        """
        Парсит JSON с несколькими попытками очистки и парсинга.
        
        Args:
            content: Строка, содержащая JSON или частично корректный JSON
            max_retries: Максимальное количество попыток парсинга
            
        Returns:
            Словарь с распарсенными данными или None, если не удалось распарсить
        """
        import copy
        
        # Первая попытка: пробуем распарсить напрямую
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Очищаем контент и пробуем снова
        cleaned_content = self._clean_json_response(content)
        
        # Попытка с очищенным контентом
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            pass

        # Попытки с разными стратегиями очистки
        strategies = [
            # Стратегия 1: удалить строки комментариев
            lambda s: re.sub(r'\s*//.*$', '', s, flags=re.MULTILINE),
            # Стратегия 2: заменить одинарные кавычки на двойные в ключах и строковых значениях
            lambda s: re.sub(r"'([^']*)':", r'"\1":', s),  # ключи
            lambda s: re.sub(r":\s*'([^']*)'", r': "\1"', s),  # значения
            # Стратегия 3: комбинация предыдущих
            lambda s: re.sub(r'\s*//.*$', '', re.sub(r"'([^']*)':", r'"\1":', s), flags=re.MULTILINE),
        ]
        
        for attempt in range(max_retries):
            for strategy in strategies:
                try:
                    processed_content = strategy(cleaned_content)
                    # Попробуем снова очистить после применения стратегии
                    processed_content = self._clean_json_response(processed_content)
                    return json.loads(processed_content)
                except (json.JSONDecodeError, TypeError):
                    continue
                    
        # Если все попытки не удались, логируем необработанный ответ для отладки
        logger.warning(f"GigaChat returned malformed JSON that could not be parsed after {max_retries} attempts.")
        logger.debug(f"Original content: {content}")
        logger.debug(f"Cleaned content: {cleaned_content}")
        
        return None

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
