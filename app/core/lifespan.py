import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Простой lifespan без фоновых задач.
    Uvicorn сам обрабатывает сигналы завершения.
    """
    logger.info("🚀 Запуск Speech Coach API")

    # Инициализация состояния
    app.state.initialized = True

    try:
        # Быстрая инициализация
        logger.info("⏳ Инициализация...")

        # Проверка доступности модулей
        try:
            from app.api.deps import get_transcriber
            transcriber = get_transcriber()
            logger.info(f"✅ Transcriber initialization: model_available={transcriber._model_available}")
        except Exception as e:
            logger.warning(f"⚠️  Transcriber initialization failed: {e}")

        # Ленивая инициализация GigaChat (при первом запросе)
        try:
            from app.core.config import settings
            if settings.gigachat_enabled:
                logger.info(
                    "🔧 GigaChat настроен, будет инициализирован при первом запросе")
        except:
            logger.debug("GigaChat не настроен")

        logger.info("✅ Приложение готово")
        yield

    except Exception as e:
        logger.error(f"💥 Ошибка при запуске: {e}")
        raise

    finally:
        logger.info("🛑 Завершение работы...")

        # Простая очистка
        try:
            # Закрытие GigaChat, если он был инициализирован
            if hasattr(app.state, 'gigachat_client'):
                try:
                    await app.state.gigachat_client.close()
                    logger.info("🔒 GigaChat закрыт")
                except Exception as e:
                    logger.debug(f"Ошибка закрытия GigaChat: {e}")
        except:
            pass

        # Минимальная очистка временных файлов
        try:
            import tempfile
            import os
            import time
            import glob

            temp_dir = tempfile.gettempdir()
            patterns = ["tmp*.mp4", "tmp*.wav", "ffmpeg*"]

            deleted = 0
            for pattern in patterns:
                for filepath in glob.glob(os.path.join(temp_dir, pattern)):
                    try:
                        # Удаляем только старые файлы (старше 1 часа)
                        if os.path.exists(filepath) and time.time() - os.path.getmtime(filepath) > 3600:
                            os.remove(filepath)
                            deleted += 1
                    except:
                        pass

            if deleted:
                logger.debug(f"🗑️  Удалено {deleted} временных файлов")
        except:
            pass

        logger.info("👋 Завершение работы выполнено")
