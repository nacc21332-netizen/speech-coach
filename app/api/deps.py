import logging
from functools import lru_cache
from typing import Optional

from app.services.audio_extractor_advanced import AdvancedFfmpegAudioExtractor
from app.services.transcriber import LocalWhisperTranscriber
from app.services.analyzer import SpeechAnalyzer
from app.services.gigachat import GigaChatClient
from app.services.pipeline import SpeechAnalysisPipeline
from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_audio_extractor() -> AdvancedFfmpegAudioExtractor:
    """Создает экстрактор аудио"""
    return AdvancedFfmpegAudioExtractor()


@lru_cache(maxsize=1)
def get_transcriber() -> LocalWhisperTranscriber:
    """Создает трансскрайбер (загружает модель при первом вызове)"""
    return LocalWhisperTranscriber()


@lru_cache(maxsize=1)
def get_analyzer() -> SpeechAnalyzer:
    """Создает анализатор речи"""
    return SpeechAnalyzer()


@lru_cache(maxsize=1)
def get_gigachat_client() -> Optional[GigaChatClient]:
    """Создает клиент GigaChat, если настроен"""
    if not settings.gigachat_enabled:
        logger.debug("GigaChat отключен в настройках")
        return None

    if not settings.gigachat_api_key:
        logger.warning("GigaChat API ключ не настроен")
        return None

    try:
        client = GigaChatClient(verify_ssl=False)
        logger.info("GigaChat клиент создан")
        return client
    except Exception as e:
        logger.error(f"Ошибка создания GigaChat клиента: {e}")
        return None


@lru_cache(maxsize=1)
def get_speech_pipeline() -> SpeechAnalysisPipeline:
    """Создает пайплайн анализа"""
    transcriber = get_transcriber()
    analyzer = get_analyzer()
    gigachat_client = get_gigachat_client()

    logger.info(f"Создание пайплайна анализа")

    return SpeechAnalysisPipeline(
        transcriber=transcriber,
        analyzer=analyzer,
        gigachat_client=gigachat_client,
    )


# Новые зависимости для расширенного анализа
@lru_cache(maxsize=1)
def get_advanced_pipeline():
    """Создает расширенный пайплайн анализа с таймингами"""
    try:
        from app.services import AdvancedSpeechAnalyzer
        from app.services.pipeline_advanced import AdvancedSpeechAnalysisPipeline

        transcriber = get_transcriber()
        analyzer = AdvancedSpeechAnalyzer()
        gigachat_client = get_gigachat_client()

        logger.info("Создание расширенного пайплайна анализа")

        return AdvancedSpeechAnalysisPipeline(
            transcriber=transcriber,
            analyzer=analyzer,
            gigachat_client=gigachat_client,
            include_timings=True
        )
    except ImportError as e:
        logger.error(f"Не удалось создать расширенный пайплайн: {e}")
        raise
