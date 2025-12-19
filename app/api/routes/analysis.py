import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from app.api.deps import get_speech_pipeline, get_advanced_pipeline
from app.models.analysis import AnalysisResult
from app.models.timed_models import TimedAnalysisResult
from app.core.exceptions import (
    FileValidationError,
    TranscriptionError,
    AnalysisError,
)

router = APIRouter(prefix="/api", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Базовый анализ видеофайла с речью",
    description="""
    Анализирует видеофайл и возвращает основные метрики качества речи.
    
    Поддерживаемые форматы: MP4, MOV, AVI, MKV, WEBM, FLV, WMV, M4V
    Максимальный размер файла: 100 MB
    
    Возвращает:
    - Основные метрики (темп речи, длительность, слова-паразиты)
    - Статистику по паузам и фразам
    - Базовые рекомендации
    - Транскрипт текста
    """,
    responses={
        200: {"description": "Анализ успешно выполнен"},
        400: {"description": "Некорректный файл или формат"},
        413: {"description": "Файл слишком большой"},
        500: {"description": "Ошибка при обработке файла"},
    }
)
async def analyze_video(
    file: UploadFile = File(...,
                            description="Видеофайл для анализа (до 100 MB)"),
    pipeline=Depends(get_speech_pipeline),
) -> AnalysisResult:
    """
    Анализирует загруженное видео и возвращает основные результаты анализа речи.
    Подходит для быстрого анализа без детализированных таймингов.
    """
    logger.info(f"Получен запрос на базовый анализ файла: {file.filename}")

    try:
        result = await pipeline.analyze_upload(file)
        logger.info(f"Базовый анализ завершен для {file.filename}")
        return result

    except FileValidationError as e:
        logger.warning(f"Ошибка валидации файла {file.filename}: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except (TranscriptionError, AnalysisError) as e:
        logger.error(f"Ошибка обработки {file.filename}: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"Неожиданная ошибка для {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера при обработке файла"
        )


@router.post(
    "/analyze/detailed",
    response_model=TimedAnalysisResult,
    summary="Детализированный анализ с таймингами",
    description="""
    Анализирует видеофайл и возвращает полные тайминги для всех элементов речи.
    
    Поддерживаемые форматы: MP4, MOV, AVI, MKV, WEBM, FLV, WMV, M4V
    Максимальный размер файла: 100 MB
    
    Возвращает:
    - Тайминги каждого слова с точными временами
    - Слова-паразиты с позициями и контекстом
    - Паузы с типом, длительностью и контекстом
    - Фразы с метриками сложности и темпа
    - Сомнительные моменты с рекомендациями
    - Данные для визуализации (активность речи, профиль интенсивности)
    - Расширенный анализ от GigaChat (если настроен)
    """,
    responses={
        200: {"description": "Детализированный анализ успешно выполнен"},
        400: {"description": "Некорректный файл или формат"},
        413: {"description": "Файл слишком большой"},
        500: {"description": "Ошибка при обработке файла"},
    }
)
async def analyze_video_detailed(
    file: UploadFile = File(...,
                            description="Видеофайл для детализированного анализа"),
    pipeline=Depends(get_advanced_pipeline),
) -> TimedAnalysisResult:
    """
    Анализирует загруженное видео и возвращает детализированные результаты
    с полными таймингами для всех элементов речи.

    Идеально для создания интерактивных временных шкал и подробных отчетов.
    """
    logger.info(f"Получен запрос на детализированный анализ файла: {file.filename}")

    try:
        result = await pipeline.analyze_with_timings(file)
        logger.info(f"Детализированный анализ завершен для {file.filename}: "
                    f"{len(result.timeline.words)} слов, "
                    f"{len(result.timeline.fillers)} слов-паразитов, "
                    f"{len(result.timeline.suspicious_moments)} проблемных моментов")
        return result

    except FileValidationError as e:
        logger.warning(f"Ошибка валидации файла {file.filename}: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except (TranscriptionError, AnalysisError) as e:
        logger.error(f"Ошибка обработки {file.filename}: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        logger.error(f"Неожиданная ошибка для {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера при обработке файла: {str(e)}"
        )


@router.post(
    "/analyze/compare",
    summary="Сравнительный анализ нескольких файлов",
    description="""
    Сравнивает несколько выступлений по ключевым метрикам.
    
    Возвращает:
    - Сравнительную таблицу метрик
    - Сильные и слабые стороны каждого выступления
    - Рекомендации по улучшению
    """,
    responses={
        200: {"description": "Сравнительный анализ выполнен"},
        400: {"description": "Некорректные файлы"},
        413: {"description": "Суммарный размер файлов слишком большой"},
        500: {"description": "Ошибка при обработке файлов"},
    }
)
async def compare_analyses(
    files: List[UploadFile] = File(...,
                                   description="Видеофайлы для сравнения (2-5 файлов, каждый до 50 MB)",
                                   max_length=5
                                   ),
    detailed: bool = False
):
    """
    Сравнивает несколько выступлений и возвращает сравнительный анализ.

    Parameters:
    - files: список видеофайлов для сравнения (2-5 файлов)
    - detailed: если True, включает детализированные тайминги в сравнение
    """
    if len(files) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Для сравнения требуется минимум 2 файла"
        )

    if len(files) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Максимум 5 файлов для сравнения"
        )

    logger.info(f"Получен запрос на сравнение {len(files)} файлов")

    # TODO: Реализовать сравнение нескольких файлов
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Функция сравнения находится в разработке. "
               "Используйте /analyze или /analyze/detailed для отдельных файлов."
    )


@router.get(
    "/analyze/formats",
    summary="Поддерживаемые форматы файлов",
    description="Возвращает список поддерживаемых форматов видеофайлов"
)
async def get_supported_formats():
    """
    Возвращает информацию о поддерживаемых форматах файлов.
    """
    return {
        "supported_formats": [
            "MP4 (.mp4) - Наиболее рекомендуемый формат",
            "MOV (.mov) - Формат Apple QuickTime",
            "AVI (.avi) - Audio Video Interleave",
            "MKV (.mkv) - Matroska Multimedia Container",
            "WEBM (.webm) - Веб-формат, основанный на Matroska",
            "FLV (.flv) - Flash Video",
            "WMV (.wmv) - Windows Media Video",
            "M4V (.m4v) - Видео iTunes"
        ],
        "recommended_format": "MP4 (.mp4)",
        "max_size_mb": 100,
        "audio_requirements": "Файл должен содержать аудиодорожку с речью",
        "notes": [
            "Файлы конвертируются в аудио формате WAV 16kHz моно",
            "Рекомендуемая длина: 1-10 минут",
            "Минимальная громкость речи: -30 dB"
        ]
    }
