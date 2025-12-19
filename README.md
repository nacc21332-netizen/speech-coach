# Speech Coach API

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/fastapi-latest-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Speech Coach — сервис для анализа публичных выступлений по загружаемым видеофайлам. Приложение извлекает аудиодорожку, выполняет локальное распознавание речи (Whisper/faster-whisper), вычисляет набор метрик качества речи и формирует структурированные рекомендации. Опционально доступен расширенный AI-анализ через интеграцию с внешними моделями.

## Ключевые возможности:
- Базовый и детализированный анализ речи (темп, паузы, слова-паразиты, тайминги)
- Расширенный AI-анализ через внешние модели
- REST API и WebUI для интерактивного использования
- Кеширование результатов и метрики мониторинга

## Документация находится в директории `docs/`. Основные разделы:
- `docs/INDEX.md` — навигация по документации
- `docs/QUICKSTART.md` — быстрый старт
- `docs/API.md` — спецификации API
- `docs/ARCHITECTURE.md` — архитектурное описание
- `docs/DEVELOPMENT.md` — инструкции для разработчиков
- `docs/CONFIGURATION.md` — параметры и переменные окружения
- `docs/DEPLOYMENT.md` — рекомендации по развертыванию

## Быстрый старт
1. Клонируйте репозиторий и создайте виртуальное окружение:

```bash
git clone https://github.com/xXxDanya2007xXx/speech-coach.git
cd speech-coach
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

После запуска сервис будет доступен по адресу `http://localhost:8000`, документация OpenAPI — по `/docs`.

## Тесты

```bash
pip install -r requirements-ci.txt
pytest tests/ -v
```

## Примеры использования API

POST-запрос для базового анализа:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@video.mp4"
```

## Конфигурация

Основные переменные окружения (см. `.env.example` для полного списка):

```env
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
MAX_FILE_SIZE=104857600
CACHE_ENABLED=true
CORS_ORIGINS=http://localhost:3000
```

Анализируемые метрики и архитектура системы описаны в `docs/ARCHITECTURE.md` и `docs/API.md`.

## Требования

- Python 3.9+
- FFmpeg (для извлечения аудио)
- Рекомендуется 4+ ГБ оперативной памяти для локального запуска моделей

## Внесение вклада

Приветствуются pull requests! Смотрите [гайд разработки](docs/DEVELOPMENT.md) для информации о стиле кода и процессе разработки.

## Поддержка

При возникновении проблем:

1. Проверьте гайд по развертыванию в разделе "Troubleshooting" (docs/DEPLOYMENT.md)
2. Откройте issue на GitHub с подробным описанием
3. Проверьте логи: `tail -f logs/app.log`

## ⚠️ Troubleshooting

### Ошибка: "The local numpy.py stub is interfering with the real numpy package"

**Причина:** В корневой директории проекта существует файл `numpy.py`, который перекрывает реальный пакет numpy.

**Решение:** Удалите файл `numpy.py` из корневой директории:
```bash
rm numpy.py
```

Затем переустановите зависимости:
```bash
pip install -r requirements.txt
```

Или используйте автоматическое исправление:
```bash
python cleanup_and_test.py
```

### Ошибка: "faster_whisper module not installed"

**Решение:** Убедитесь что установлены все зависимости:
```bash
pip install faster-whisper ctranslate2 onnxruntime
```

Проверьте импорты:
```bash
python test_imports.py
```

---

**Последнее обновление:** December 19, 2025  
**Версия:** 1.0.0  
**Статус:** Production Ready

Проект распространяется под лицензией MIT. Смотрите [LICENSE](LICENSE) для деталей.
