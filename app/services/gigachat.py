import os
import json
import re
import logging
import uuid
import time
from typing import Optional, Dict, Any, List
from app.services.cache import AnalysisCache
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
                        "content": """Ты опытный тренер по ораторскому искусству и публичным выступлениям. Твоя задача - дать глубокий и профессиональный анализ речи на основе предоставленных метрик и транскрипта. 

Ты должен проанализировать:
1. Структуру и логичность выступления
2. Эмоциональную окраску и убедительность
3. Взаимодействие с аудиторией (по паузам, темпу речи)
4. Использование профессиональной лексики
5. Темп речи и ритмичность
6. Наличие слов-паразитов и их влияние на восприятие
7. Длину фраз и ясность выражения мыслей
8. Общее впечатление от выступления

Формат ответа: строго JSON без дополнительного текста. Все поля обязательны к заполнению, даже если данных недостаточно - дай лучшую оценку на основе доступной информации.

Формат JSON:
{
  "overall_assessment": "Общая оценка выступления: сильные и слабые стороны, уровень подготовки, общее впечатление",
  "strengths": [
    "Первая сильная сторона с конкретным примером из выступления",
    "Вторая сильная сторона с конкретным примером из выступления"
  ],
  "areas_for_improvement": [
    "Первая зона роста с конкретным указанием проблемы",
    "Вторая зона роста с конкретным указанием проблемы"
  ],
  "detailed_recommendations": [
    "Конкретная рекомендация по улучшению с объяснением",
    "Конкретная рекомендация по улучшению с объяснением"
  ],
  "key_insights": [
    "Ключевой инсайт о стиле речи",
    "Ключевой инсайт о взаимодействии с аудиторией"
  ],
  "confidence_score": "Число от 0 до 1, отражающее уверенность в анализе на основе полноты данных"
}"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": min(self.max_tokens, 2000),  # Увеличиваем для более подробного анализа
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
                # Валидируем обязательные поля
                required_fields = ["overall_assessment", "strengths", "areas_for_improvement",
                                   "detailed_recommendations", "key_insights", "confidence_score"]

                for field in required_fields:
                    if field not in parsed_content or parsed_content[field] is None:
                        if field == "confidence_score":
                            parsed_content[field] = 0.5
                        else:
                            parsed_content[field] = ""
                
                # Убедимся, что overall_assessment не пустой, если другие поля заполнены
                if not parsed_content["overall_assessment"].strip() and (
                    parsed_content["strengths"] or 
                    parsed_content["areas_for_improvement"] or 
                    parsed_content["detailed_recommendations"] or 
                    parsed_content["key_insights"]
                ):
                    parsed_content["overall_assessment"] = "Анализ выступления показал следующие результаты"

                return GigaChatAnalysis(**parsed_content)
            else:
                logger.warning("Failed to parse GigaChat response after retries")
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
                pauses_info += f"- {pause['duration']:.1f} сек (с {pause['start']:.1f} по {pause['end']:.1f})\n"

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

    async def classify_fillers_context(self, contexts: List[Dict[str, Any]], cache: Optional[AnalysisCache] = None) -> List[Dict[str, Any]]:
        """Classify each candidate filler word in context using LLM with caching and retry/backoff.
        Returns enriched list of contexts with `is_filler_context` and `score`.
        """
        if not settings.llm_fillers_enabled:
            return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]

        if not self._access_token:
            try:
                await self.authenticate()
            except GigaChatError as e:
                logger.warning(f"Failed to authenticate for filler classification: {e}")
                return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]

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
                results.append(dict(**c, is_filler_context=cached.get("is_filler", False), score=cached.get("confidence", 0.0)))
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

        user_prompt = f"Оцени, являются ли слова в каждом пункте явными словами-паразитами в данном контексте. Верни JSON-массив одинаковой длины с объектами: {{index: N, is_filler: true|false, confidence: 0..1}}\n\n{items_block}"

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
                    return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]
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
            return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]

        try:
            body = response.json()
            if not body.get("choices"):
                return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            enriched_map = {}
            for r in parsed:
                idx = r.get("index", None)
                if idx is None:
                    continue
                qpos = idx - 1
                if qpos < 0 or qpos >= len(to_query):
                    continue
                key, original = to_query[qpos]
                enriched = dict(**original, is_filler_context=bool(r.get("is_filler", False)), score=float(r.get("confidence", 0.0)))
                enriched_map[key] = enriched

            final_results = []
            final_results.extend(results)
            for key, c in to_query:
                enriched = enriched_map.get(key, dict(**c, is_filler_context=False, score=0.0))
                if cache is not None:
                    try:
                        cache.set_by_key(key, {"is_filler": enriched.get("is_filler_context", False), "confidence": enriched.get("score", 0.0)})
                    except Exception:
                        pass
                final_results.append(enriched)

            # Reassemble in the original input order
            ordered = [None] * len(contexts)
            r_i = 0
            for i in range(len(contexts)):
                ordered[i] = final_results[r_i]
                r_i += 1
            return ordered
        except Exception as e:
            logger.warning(f"Failed to parse filler classification response: {e}")
            return [dict(**c, is_filler_context=False, score=0.0) for c in contexts]

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

            # Убедимся, что токен аутентификации действителен
            try:
                await self.authenticate()
            except GigaChatError as e:
                logger.warning(f"Failed to authenticate with GigaChat: {e}")
                processing_time = time.time() - start_time
                return self._create_error_response(f"Failed to authenticate: {str(e)}", processing_time)
            
            # Проверяем, что токен действительно установлен после аутентификации
            if not self._access_token:
                logger.error("Failed to obtain access token for detailed analysis request")
                processing_time = time.time() - start_time
                return self._create_error_response("Failed to authenticate", processing_time)

            chat_url = f"{self.api_url}/chat/completions"
            request_data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """Ты опытный тренер по ораторскому искусству и публичным выступлениям. Твоя задача - дать глубокий и профессиональный анализ речи на основе предоставленных метрик, транскрипта и точных временных меток.

Ты должен проанализировать:
1. Структуру и логичность выступления
2. Эмоциональную окраску и убедительность
3. Взаимодействие с аудиторией (по паузам, темпу речи)
4. Использование профессиональной лексики
5. Темп речи и ритмичность
6. Наличие слов-паразитов и их влияние на восприятие
7. Длину фраз и ясность выражения мыслей
8. Общее впечатление от выступления
9. Привязывай все рекомендации к конкретным секундам выступления
10. Выявляй временные паттерны и закономерности
11. Определяй критические моменты (поворотные точки, кульминации)
12. Анализируй стиль речи и его уместность
13. Оценивай предполагаемую вовлеченность аудитории по времени
14. Составь временную шкалу улучшений с упражнениями

Формат ответа: строго JSON без дополнительного текста. Все поля обязательны к заполнению, даже если данных недостаточно - дай лучшую оценку на основе доступной информации.

Формат JSON:
{
  "overall_assessment": "Общая оценка выступления: сильные и слабые стороны, уровень подготовки, общее впечатление",
  "strengths": [
    "Первая сильная сторона с конкретным примером из выступления",
    "Вторая сильная сторона с конкретным примером из выступления"
  ],
  "areas_for_improvement": [
    "Первая зона роста с конкретным указанием проблемы",
    "Вторая зона роста с конкретным указанием проблемы"
  ],
  "detailed_recommendations": [
    "Конкретная рекомендация по улучшению с объяснением",
    "Конкретная рекомендация по улучшению с объяснением"
  ],
  "key_insights": [
    "Ключевой инсайт о стиле речи",
    "Ключевой инсайт о взаимодействии с аудиторией"
  ],
  "confidence_score": "Число от 0 до 1, отражающее уверенность в анализе на основе полноты данных",
  "time_based_analysis": [
    {
      "time_range": "Временной диапазон в секундах",
      "observation": "Что происходит в этот момент",
      "recommendation": "Рекомендации для этого временного диапазона"
    }
  ],
  "temporal_patterns": [
    {
      "pattern": "Обнаруженный паттерн",
      "time_instances": [секунды],
      "description": "Описание паттерна"
    }
  ],
  "improvement_timeline": [
    {
      "time_marker": "Время в секундах",
      "improvement_area": "Область для улучшения",
      "exercise": "Упражнение для улучшения"
    }
  ],
  "critical_moments": [
    {
      "time": "Время в секундах",
      "type": "Тип критического момента",
      "description": "Описание критического момента"
    }
  ]
}"""
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

            try:
                response = await self.client.post(chat_url, json=request_data, headers=headers)
            except httpx.ConnectError as e:
                logger.error(f"GigaChat API connection error: {e}")
                logger.error(f"Could not connect to GigaChat API at {chat_url}")
                processing_time = time.time() - start_time
                return self._create_error_response(f"Connection error: {str(e)}", processing_time)

            if response.status_code != 200:
                logger.error(f"GigaChat API error {response.status_code}: {response.text}")
                processing_time = time.time() - start_time
                return self._create_error_response("GigaChat API error", processing_time)

            result = response.json()

            if "choices" not in result or len(result["choices"]) == 0:
                logger.error("No choices in GigaChat response")
                processing_time = time.time() - start_time
                return self._create_error_response("No choices in response", processing_time)

            content = result["choices"][0]["message"]["content"]
            processing_time = time.time() - start_time

            logger.info(f"GigaChat detailed analysis received in {processing_time:.1f} seconds")

            # Пробуем распарсить JSON с несколькими попытками
            parsed_content = self._parse_json_with_retries(content)
            
            if parsed_content is not None:
                parsed_content["processing_time_sec"] = processing_time
                return parsed_content
            else:
                logger.warning("Failed to parse GigaChat detailed response after retries")
                logger.debug(f"Raw content: {content[:2000]}...")
                processing_time = time.time() - start_time
                return self._create_error_response("JSON parse error after retries", processing_time)

        except httpx.ConnectError as e:
            logger.error(f"GigaChat API connection failed: {e}")
            processing_time = time.time() - start_time
            return self._create_error_response(f"Connection error: {str(e)}", processing_time)
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
