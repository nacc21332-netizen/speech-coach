"""
Расширенные методы GigaChat для работы с таймингами.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def create_enhanced_gigachat_analysis(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Создает расширенный анализ GigaChat из ответа API.

    Args:
        response_data: Словарь с данными от GigaChat API

    Returns:
        Словарь с расширенным анализом
    """
    try:
        # Базовые поля
        result = {
            "overall_assessment": response_data.get("overall_assessment", ""),
            "strengths": response_data.get("strengths", []),
            "areas_for_improvement": response_data.get("areas_for_improvement", []),
            "detailed_recommendations": response_data.get("detailed_recommendations", []),
            "key_insights": response_data.get("key_insights", []),
            "confidence_score": response_data.get("confidence_score", 0.0),
        }

        # Временные данные
        if "time_based_analysis" in response_data:
            result["time_based_analysis"] = response_data["time_based_analysis"]

        if "temporal_patterns" in response_data:
            result["temporal_patterns"] = response_data["temporal_patterns"]

        if "improvement_timeline" in response_data:
            result["improvement_timeline"] = response_data["improvement_timeline"]

        if "critical_moments" in response_data:
            result["critical_moments"] = response_data["critical_moments"]

        # Дополнительные поля
        if "speech_style" in response_data:
            result["speech_style"] = response_data["speech_style"]

        if "audience_engagement" in response_data:
            result["audience_engagement"] = response_data["audience_engagement"]

        # Метаданные
        result["metadata"] = {
            "has_timed_analysis": "time_based_analysis" in response_data,
            "has_patterns": "temporal_patterns" in response_data,
            "has_improvement_plan": "improvement_timeline" in response_data,
            "processing_time_sec": response_data.get("processing_time_sec"),
            "version": "2.0"
        }

        return result

    except Exception as e:
        logger.error(f"Error creating enhanced GigaChat analysis: {e}")

        # Возвращаем базовую структуру в случае ошибки
        return {
            "overall_assessment": "Анализ выполнен с ограниченными возможностями",
            "strengths": [],
            "areas_for_improvement": ["Ошибка обработки расширенного анализа"],
            "detailed_recommendations": ["Попробуйте базовый анализ"],
            "key_insights": [f"Ошибка: {str(e)}"],
            "confidence_score": 0.1,
            "metadata": {
                "has_timed_analysis": False,
                "error": str(e)
            }
        }


def prepare_timed_result_for_gigachat(timed_result: Any) -> Dict[str, Any]:
    """
    Подготавливает результат с таймингами для отправки в GigaChat.

    Args:
        timed_result: Объект TimedAnalysisResult или словарь

    Returns:
        Словарь с данными для GigaChat
    """
    try:
        # Если это объект Pydantic, преобразуем в словарь
        if hasattr(timed_result, "dict"):
            data = timed_result.dict()
        else:
            data = dict(timed_result)

        # Убедимся, что timeline есть
        timeline = data.get("timeline", {})
        if not timeline:
            timeline = {
                "words": [],
                "fillers": [],
                "pauses": [],
                "phrases": [],
                "suspicious_moments": []
            }

        # Ограничиваем размер данных для отправки
        result = {
            "duration_sec": data.get("duration_sec", 0),
            "speaking_time_sec": data.get("speaking_time_sec", 0),
            "speaking_ratio": data.get("speaking_ratio", 0),
            "words_total": data.get("words_total", 0),
            "words_per_minute": data.get("words_per_minute", 0),
            # Ограничиваем транскрипт
            "transcript": data.get("transcript", "")[:3000],
            "timeline": {
                # Ограничиваем количество
                "fillers": timeline.get("fillers", [])[:50],
                "pauses": [p for p in timeline.get("pauses", []) if p.get("duration", 0) > 1.0][:30],
                "phrases": timeline.get("phrases", [])[:20],
                "suspicious_moments": timeline.get("suspicious_moments", [])[:20]
            }
        }

        return result

    except Exception as e:
        logger.error(f"Error preparing timed result for GigaChat: {e}")

        # Возвращаем минимальные данные
        return {
            "duration_sec": 0,
            "speaking_time_sec": 0,
            "speaking_ratio": 0,
            "words_total": 0,
            "words_per_minute": 0,
            "transcript": "",
            "timeline": {
                "fillers": [],
                "pauses": [],
                "phrases": [],
                "suspicious_moments": []
            }
        }
