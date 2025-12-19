# 🔧 Практический гайд: Улучшение Кэша, Логирования и Асинхронности

**Целевой уровень**: Production-ready
**Время внедрения**: 4-6 часов
**Сложность**: Средняя

---

## БЫСТРЫЙ СТАРТ (15 минут)

### Что сразу дает результат

```bash
# 1. Установка необходимых зависимостей (для основных улучшений)
pip install cachetools pythonjsonlogger

# 2. Опционально (для Redis - лучший вариант)
pip install redis

# 3. Опционально (для circuit breaker)
pip install pybreaker
```

---

## 1️⃣ УЛУЧШЕНИЕ КЭШИРОВАНИЯ (ПРИОРИТЕТ 1)

### Вариант A: Быстрое решение (без внешних зависимостей)

Добавьте in-memory LRU кэш перед диском:

```python
# app/services/cache_manager.py - НОВЫЙ ФАЙЛ

import asyncio
import logging
from typing import Any, Optional
from cachetools import TTLCache
from pathlib import Path

logger = logging.getLogger(__name__)

class TwoLevelCache:
    """Двухуровневый кэш: память (горячие данные) + диск (холодные)"""
    
    def __init__(
        self,
        disk_cache=None,  # AnalysisCache instance
        memory_maxsize: int = 100,
        ttl_seconds: int = 3600
    ):
        # L1: In-memory кэш для горячих данных
        self.memory = TTLCache(maxsize=memory_maxsize, ttl=ttl_seconds)
        
        # L2: Диск кэш для холодных данных
        self.disk = disk_cache
        
        # Статистика
        self.hits_memory = 0
        self.hits_disk = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Получить с приоритетом память → диск"""
        
        # Попытка L1 (0.1ms)
        if key in self.memory:
            self.hits_memory += 1
            logger.debug(f"Cache L1 hit: {key}")
            return self.memory[key]
        
        # Попытка L2 (10-50ms)
        if self.disk:
            value = await asyncio.to_thread(self.disk.get_by_key, key)
            if value is not None:
                # Переместить в L1 для следующего раза
                self.memory[key] = value
                self.hits_disk += 1
                logger.debug(f"Cache L2 hit: {key}")
                return value
        
        self.misses += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any) -> None:
        """Сохранить в оба уровня кэша"""
        self.memory[key] = value
        
        if self.disk:
            await asyncio.to_thread(self.disk.set_by_key, key, value)
    
    def stats(self) -> dict:
        """Статистика эффективности кэша"""
        total = self.hits_memory + self.hits_disk + self.misses
        hit_rate = (self.hits_memory + self.hits_disk) / total if total > 0 else 0
        
        return {
            "hits_memory": self.hits_memory,
            "hits_disk": self.hits_disk,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": f"{hit_rate*100:.1f}%",
            "memory_size": len(self.memory),
            "memory_maxsize": self.memory.maxsize,
        }
    
    async def clear(self) -> None:
        """Очистить оба уровня"""
        self.memory.clear()
        if self.disk:
            await asyncio.to_thread(self.disk.clear_old)
```

#### Интеграция в код

```python
# app/api/deps.py - ОБНОВИТЬ

from app.services.cache_manager import TwoLevelCache
from app.services.cache import AnalysisCache
from pathlib import Path

@lru_cache(maxsize=1)
def get_cache_manager() -> TwoLevelCache:
    """Создает двухуровневый кэш менеджер"""
    disk_cache = AnalysisCache(
        cache_dir=Path(settings.cache_dir),
        ttl_seconds=settings.cache_ttl
    )
    return TwoLevelCache(
        disk_cache=disk_cache,
        memory_maxsize=100,  # До 100 анализов в памяти
        ttl_seconds=settings.cache_ttl
    )
```

#### Добавить endpoint для мониторинга кэша

```python
# app/api/routes/health.py - ДОБАВИТЬ

from app.api.deps import get_cache_manager

@router.get("/stats/cache")
async def cache_stats(cache_manager = Depends(get_cache_manager)):
    """Получить статистику кэша"""
    return cache_manager.stats()

# Использование:
# GET /stats/cache
# Ответ:
# {
#   "hits_memory": 45,
#   "hits_disk": 12,
#   "misses": 8,
#   "hit_rate": "87.7%",
#   "memory_size": 23
# }
```

---

### Вариант B: Профессиональное решение (с Redis)

Если у вас есть Redis (рекомендуется для production):

```python
# app/services/cache_manager.py - ВАРИАНТ С REDIS

import asyncio
import logging
from typing import Any, Optional
from cachetools import TTLCache
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
import json

logger = logging.getLogger(__name__)

class RedisCache:
    """Трёхуровневый кэш: память + Redis + диск"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        disk_cache=None,
        memory_maxsize: int = 50,
        ttl_seconds: int = 3600
    ):
        # L1: In-memory
        self.memory = TTLCache(maxsize=memory_maxsize, ttl=ttl_seconds)
        
        # L2: Redis (если доступен)
        try:
            self.redis = Redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()  # Проверка соединения
            self.redis_available = True
            logger.info(f"Redis connected: {redis_url}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis = None
            self.redis_available = False
        
        # L3: Диск
        self.disk = disk_cache
        self.ttl_seconds = ttl_seconds
        
        # Статистика
        self.hits_l1 = self.hits_l2 = self.hits_l3 = self.misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """L1 (0.1ms) → L2 (1-5ms) → L3 (10-50ms)"""
        
        # L1: Память
        if key in self.memory:
            self.hits_l1 += 1
            return self.memory[key]
        
        # L2: Redis
        if self.redis_available:
            try:
                value_json = await asyncio.to_thread(self.redis.get, key)
                if value_json:
                    import pickle
                    value = pickle.loads(json.loads(value_json))
                    self.memory[key] = value  # Переместить в L1
                    self.hits_l2 += 1
                    return value
            except Exception as e:
                logger.debug(f"Redis get error: {e}")
        
        # L3: Диск
        if self.disk:
            try:
                value = await asyncio.to_thread(self.disk.get_by_key, key)
                if value:
                    self.memory[key] = value
                    if self.redis_available:
                        # Переместить в Redis для ускорения
                        await self._set_redis(key, value)
                    self.hits_l3 += 1
                    return value
            except Exception as e:
                logger.debug(f"Disk get error: {e}")
        
        self.misses += 1
        return None
    
    async def set(self, key: str, value: Any) -> None:
        """Сохранить во все уровни"""
        self.memory[key] = value
        
        if self.redis_available:
            await self._set_redis(key, value)
        
        if self.disk:
            await asyncio.to_thread(self.disk.set_by_key, key, value)
    
    async def _set_redis(self, key: str, value: Any) -> None:
        """Вспомогательный метод для сохранения в Redis"""
        try:
            import pickle
            value_json = json.dumps(pickle.dumps(value).decode('latin1'))
            await asyncio.to_thread(
                self.redis.setex,
                key,
                self.ttl_seconds,
                value_json
            )
        except Exception as e:
            logger.debug(f"Redis set error: {e}")
    
    def stats(self) -> dict:
        """Расширенная статистика"""
        total = self.hits_l1 + self.hits_l2 + self.hits_l3 + self.misses
        
        return {
            "l1_memory_hits": self.hits_l1,
            "l2_redis_hits": self.hits_l2,
            "l3_disk_hits": self.hits_l3,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{(total-self.misses)/total*100:.1f}%" if total > 0 else "0%",
            "memory_size": len(self.memory),
            "redis_available": self.redis_available,
        }
```

**Конфиг Redis в `.env`**:
```env
# Если используется Redis
REDIS_URL=redis://localhost:6379/0

# Параметры кэша
CACHE_TTL=3600
CACHE_DIR=cache/analysis
```

---

## 2️⃣ JSON ЛОГИРОВАНИЕ (ПРИОРИТЕТ 2)

### Шаг 1: Установка

```bash
pip install pythonjsonlogger
```

### Шаг 2: Обновить logging config

```python
# app/core/logging_config.py - ОБНОВИТЬ

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import coloredlogs
from pythonjsonlogger import jsonlogger
import uuid
from contextvars import ContextVar

# Контекст для Request ID
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

def get_request_id() -> str:
    """Получить текущий Request ID"""
    return request_id_var.get()

def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    json_logs: bool = True  # ← НОВЫЙ ПАРАМЕТР
):
    """Настраивает логирование с поддержкой JSON"""
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Форматер для консоли (текст с цветами)
    console_formatter = coloredlogs.ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    
    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    
    # Консоль (текст)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Файл (JSON если включено)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if json_logs:
            # JSON форматер для машинной обработки
            json_formatter = jsonlogger.JsonFormatter()
            file_handler.setFormatter(json_formatter)
        else:
            # Текстовый форматер
            text_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(text_formatter)
        
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    # Подавить шумные библиотеки
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging configured: {log_level} (JSON={json_logs})")
```

### Шаг 3: Добавить Request ID middleware

```python
# app/main.py - ДОБАВИТЬ

import uuid
from contextvars import ContextVar
from fastapi.requests import Request

# Импортируем функцию для установки Request ID
from app.core.logging_config import request_id_var

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Добавляет уникальный Request ID для каждого запроса"""
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    
    response = await call_next(request)
    
    # Добавляем в headers для отладки
    response.headers["X-Request-ID"] = request_id
    
    return response
```

### Шаг 4: Использование в коде

```python
# Где угодно в коде
import logging
from app.core.logging_config import get_request_id

logger = logging.getLogger(__name__)

async def analyze_speech(...):
    request_id = get_request_id()
    logger.info(f"Starting analysis", extra={
        "request_id": request_id,
        "file_size": file_size,
        "duration": duration
    })
    
    # Результаты логов с автоматическим request_id
```

### Результат в логах

```json
// logs/app.log (JSON формат)
{"asctime": "2025-12-19 14:30:45", "name": "app.api.routes", "levelname": "INFO", "message": "Starting analysis", "request_id": "abc-123-def", "file_size": 5000000, "duration": 30.5}
{"asctime": "2025-12-19 14:31:15", "name": "app.services", "levelname": "INFO", "message": "Analysis completed", "request_id": "abc-123-def"}

// stdout (текст с цветами)
14:30:45 - app.api.routes - INFO - Starting analysis
14:31:15 - app.services - INFO - Analysis completed
```

---

## 3️⃣ CIRCUIT BREAKER для GigaChat (ОПЦИОНАЛЬНО)

### Установка

```bash
pip install pybreaker
```

### Реализация

```python
# app/services/gigachat.py - ОБНОВИТЬ

from pybreaker import CircuitBreaker
import logging

logger = logging.getLogger(__name__)

class GigaChatClient:
    def __init__(self, verify_ssl: Optional[bool] = None):
        # ... существующий код ...
        
        # Circuit breaker: отключает API после 5 ошибок на 60 сек
        self.breaker = CircuitBreaker(
            fail_max=5,  # Отключить после 5 ошибок
            reset_timeout=60,  # На 60 секунд
            listeners=[
                self._on_breaker_change
            ]
        )
    
    def _on_breaker_change(self, event):
        """Callback при изменении состояния circuit breaker"""
        logger.warning(f"Circuit breaker event: {event}")
        if event == "open":
            logger.error("GigaChat API circuit opened - too many failures")
        elif event == "close":
            logger.info("GigaChat API circuit closed - API recovered")
    
    async def analyze_speech(self, analysis_result: AnalysisResult):
        """Анализ с защитой от каскадных сбоев"""
        try:
            # Проверка circuit breaker ПЕРЕД запросом
            if not self.breaker.closed:
                logger.warning("Circuit breaker is open, using fallback")
                return self._create_fallback_analysis(
                    "API temporarily unavailable"
                )
            
            # Обычный запрос
            response = await self.client.post(...)
            self.breaker.success()  # Отметить успех
            return response
            
        except Exception as e:
            self.breaker.fail()  # Отметить ошибку
            logger.error(f"Analysis failed: {e}")
            return self._create_fallback_analysis(str(e))
```

---

## 4️⃣ SEMAPHORE для ограничения параллелизма

```python
# app/services/pipeline.py - ДОБАВИТЬ

import asyncio

class SpeechAnalysisPipeline:
    def __init__(self, max_parallel: int = 3):
        # Ограничить до 3 одновременных анализов
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.active_count = 0
        self.max_parallel = max_parallel
    
    async def analyze(self, file, ...):
        """Анализ с ограничением параллелизма"""
        async with self.semaphore:
            self.active_count += 1
            logger.info(f"Active analyses: {self.active_count}/{self.max_parallel}")
            
            try:
                return await self._do_analyze(file, ...)
            finally:
                self.active_count -= 1
    
    def status(self) -> dict:
        """Статус нагрузки"""
        return {
            "active": self.active_count,
            "max_parallel": self.max_parallel,
            "queue_length": self.semaphore._value,
        }
```

---

## 🧪 ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ

### Базовый тест кэша

```python
# tests/test_improved_cache.py

import pytest
from app.services.cache_manager import TwoLevelCache
from app.services.cache import AnalysisCache
from pathlib import Path
import asyncio

@pytest.mark.asyncio
async def test_two_level_cache():
    """Тест двухуровневого кэша"""
    disk = AnalysisCache(Path("cache/test"), ttl_seconds=3600)
    cache = TwoLevelCache(disk_cache=disk, memory_maxsize=10)
    
    # Первый запрос → misses
    value1 = await cache.get("key1")
    assert value1 is None
    assert cache.misses == 1
    
    # Сохранить
    test_data = {"result": "analysis"}
    await cache.set("key1", test_data)
    
    # Второй запрос → L1 hit (память)
    value2 = await cache.get("key1")
    assert value2 == test_data
    assert cache.hits_memory == 1
    
    # Stats
    stats = cache.stats()
    assert "87.5%" in stats["hit_rate"]  # 1 hit из 2 requests было раньше
```

### Тест JSON логирования

```python
# tests/test_json_logging.py

import logging
import json
from io import StringIO
from pythonjsonlogger import jsonlogger

def test_json_logging():
    """Тест JSON логирования"""
    logger = logging.getLogger("test")
    
    # JSON handler
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(jsonlogger.JsonFormatter())
    logger.addHandler(handler)
    
    # Логируем
    logger.info("Test message", extra={"user_id": 123})
    
    # Проверяем JSON
    log_line = stream.getvalue().strip()
    log_dict = json.loads(log_line)
    
    assert log_dict["message"] == "Test message"
    assert log_dict["user_id"] == 123
```

---

## 📊 МЕТРИКИ ДО И ПОСЛЕ

```
                        | ДО       | ПОСЛЕ    | УЛУЧШЕНИЕ
|----------------------|----------|----------|----------
| Время кэш хита      | 10-50ms  | 0.1-1ms  | 50-500x
| JSON логирование    | ❌ Нет   | ✅ Да    | Парсинг
| Request трейсинг    | ❌ Нет   | ✅ Да    | Отладка
| Memory L1 кэш       | ❌ Нет   | ✅ Да    | +100x
| Circuit breaker     | ❌ Нет   | ✅ Да    | Надежность
```

---

## 🚀 РАЗВЕРТЫВАНИЕ

### В production

```bash
# 1. Установить зависимости
pip install -r requirements.txt  # Уже включены новые

# 2. Стартовый скрипт с Redis (опционально)
docker-compose up -d redis

# 3. Запуск с JSON логами
LOG_LEVEL=INFO REDIS_URL=redis://localhost:6379 python -m uvicorn app.main:app

# 4. Мониторить кэш
curl http://localhost:8000/stats/cache
```

---

## 📝 Чеклист внедрения

- [ ] Установлены зависимости (cachetools, pythonjsonlogger)
- [ ] Добавлен TwoLevelCache или RedisCache
- [ ] JSON логирование настроено
- [ ] Request ID middleware добавлен
- [ ] Circuit breaker для GigaChat (опционально)
- [ ] Semaphore для параллелизма (опционально)
- [ ] Endpoint /stats/cache добавлен
- [ ] Тесты написаны
- [ ] Локально протестировано
- [ ] В production задеплоено

