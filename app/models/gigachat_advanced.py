"""
Расширенные модели для анализа GigaChat с поддержкой таймингов.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class TimeBasedAnalysisItem(BaseModel):
    """Элемент анализа, привязанный ко времени"""
    timestamp: float = Field(description="Метка времени в секундах")
    type: str = Field(
        description="Тип: problem|strength|suggestion|question|emphasis")
    title: str = Field(description="Краткое название (до 50 символов)")
    description: str = Field(description="Подробное описание")
    suggestion: Optional[str] = Field(
        None, description="Конкретная рекомендация")
    context: str = Field(description="Контекст (окрестные слова)")
    severity: Optional[str] = Field(
        None, description="Серьезность: low|medium|high")
    confidence: float = Field(
        0.8, ge=0.0, le=1.0, description="Уверенность анализа")
    tags: List[str] = Field(default_factory=list,
                            description="Теги для категоризации")
    affected_words: List[str] = Field(
        default_factory=list, description="Затронутые слова")
    improvement_potential: float = Field(
        0.5, ge=0.0, le=1.0, description="Потенциал улучшения")


class TemporalPattern(BaseModel):
    """Временной паттерн в речи"""
    pattern: str = Field(description="Описание паттерна")
    time_range: str = Field(
        description="Временной диапазон (например, '0-30 секунд')")
    description: str = Field(description="Подробное описание паттерна")
    recommendation: str = Field(description="Рекомендация по паттерну")
    occurrences: int = Field(description="Количество повторений паттерна")
    confidence: float = Field(
        0.8, ge=0.0, le=1.0, description="Уверенность в определении паттерна")
    examples: List[str] = Field(
        default_factory=list, description="Примеры проявления")
    impact: Literal["low", "medium", "high"] = Field(
        "medium", description="Влияние на восприятие")


class ImprovementTimelineItem(BaseModel):
    """Элемент временной шкалы улучшений"""
    start_time: float = Field(description="Начало интервала в секундах")
    end_time: float = Field(description="Конец интервала в секундах")
    focus_area: str = Field(description="Область улучшения")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        "medium", description="Приоритет улучшения")
    exercises: List[str] = Field(description="Упражнения для этого интервала")
    expected_improvement: str = Field(description="Ожидаемое улучшение")
    time_required_min: int = Field(description="Требуемое время в минутах")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        "medium", description="Сложность упражнений")
    success_metrics: List[str] = Field(
        default_factory=list, description="Метрики успеха")


class CriticalMoment(BaseModel):
    """Критический момент выступления"""
    timestamp: float = Field(description="Время момента в секундах")
    type: Literal["turning_point", "climax", "weak_point", "engagement_drop",
                  "key_message"] = Field("key_message", description="Тип момента")
    description: str = Field(description="Описание момента")
    impact: float = Field(0.5, ge=0.0, le=1.0,
                          description="Влияние на восприятие (0-1)")
    audience_reaction: Optional[str] = Field(
        None, description="Предполагаемая реакция аудитории")
    alternative_approaches: List[str] = Field(
        default_factory=list, description="Альтернативные подходы")
    lessons_learned: List[str] = Field(
        default_factory=list, description="Извлеченные уроки")


class SpeechStyleAnalysis(BaseModel):
    """Анализ стиля речи"""
    style: str = Field(description="Определенный стиль речи")
    confidence: float = Field(
        0.8, ge=0.0, le=1.0, description="Уверенность в определении")
    characteristics: List[str] = Field(description="Характеристики стиля")
    suitability: float = Field(
        0.5, ge=0.0, le=1.0, description="Подходит ли стиль для контекста")
    recommendations: List[str] = Field(
        default_factory=list, description="Рекомендации по стилю")
    examples_from_speech: List[str] = Field(
        default_factory=list, description="Примеры из выступления")


class AudienceEngagementAnalysis(BaseModel):
    """Анализ вовлеченности аудитории"""
    overall_engagement: float = Field(
        0.5, ge=0.0, le=1.0, description="Общая вовлеченность (0-1)")
    engagement_timeline: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Временная шкала вовлеченности"
    )
    peak_engagement_times: List[float] = Field(
        default_factory=list,
        description="Время пиков вовлеченности"
    )
    drop_times: List[float] = Field(
        default_factory=list,
        description="Время спадов вовлеченности"
    )
    engagement_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Факторы, влияющие на вовлеченность"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Предложения по улучшению вовлеченности"
    )


class EnhancedGigaChatAnalysis(BaseModel):
    """Расширенный анализ от GigaChat с поддержкой таймингов"""
    # Основные поля (совместимость с базовой моделью)
    overall_assessment: str = Field(
        description="Общая оценка выступления (1-2 абзаца)")
    strengths: List[str] = Field(description="Сильные стороны выступления")
    areas_for_improvement: List[str] = Field(description="Основные зоны роста")
    detailed_recommendations: List[str] = Field(
        description="Конкретные рекомендации")
    key_insights: List[str] = Field(description="Ключевые инсайты из анализа")
    confidence_score: float = Field(
        0.8, ge=0.0, le=1.0, description="Уверенность анализа (0-1)")

    # Новые поля для работы с таймингами
    time_based_analysis: List[TimeBasedAnalysisItem] = Field(
        default_factory=list,
        description="Анализ, привязанный к конкретным временным меткам"
    )
    temporal_patterns: List[TemporalPattern] = Field(
        default_factory=list,
        description="Выявленные временные паттерны в речи"
    )
    improvement_timeline: List[ImprovementTimelineItem] = Field(
        default_factory=list,
        description="Временная шкала рекомендуемых улучшений"
    )
    critical_moments: List[CriticalMoment] = Field(
        default_factory=list,
        description="Критические моменты выступления"
    )
    speech_style: Optional[SpeechStyleAnalysis] = Field(
        None,
        description="Анализ стиля речи"
    )
    audience_engagement: Optional[AudienceEngagementAnalysis] = Field(
        None,
        description="Анализ вовлеченности аудитории"
    )

    # Метаданные
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Время проведения анализа"
    )
    model_version: str = Field(
        "GigaChat", description="Версия модели GigaChat")
    processing_time_sec: Optional[float] = Field(
        None, description="Время обработки в секундах")

    # Методы для удобства
    def get_time_based_summary(self, time_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Получить сводку по временному диапазону"""
        if time_range:
            start, end = time_range
            items = [
                item for item in self.time_based_analysis if start <= item.timestamp <= end]
        else:
            items = self.time_based_analysis

        problems = [item for item in items if item.type == "problem"]
        strengths = [item for item in items if item.type == "strength"]
        suggestions = [item for item in items if item.type == "suggestion"]

        return {
            "total_items": len(items),
            "problems": len(problems),
            "strengths": len(strengths),
            "suggestions": len(suggestions),
            "most_common_problem": max(
                set([p.title for p in problems]),
                key=[p.title for p in problems].count
            ) if problems else None,
            "critical_problems": [p for p in problems if p.severity == "high"],
            "top_suggestions": suggestions[:3]
        }

    def get_improvement_plan(self) -> Dict[str, Any]:
        """Получить план улучшений"""
        return {
            "total_areas": len(set(item.focus_area for item in self.improvement_timeline)),
            "total_exercises": sum(len(item.exercises) for item in self.improvement_timeline),
            "total_time_min": sum(item.time_required_min for item in self.improvement_timeline),
            "priority_areas": [
                {
                    "area": item.focus_area,
                    "priority": item.priority,
                    "time_required": item.time_required_min
                }
                for item in self.improvement_timeline
                if item.priority in ["high", "critical"]
            ],
            "quick_wins": [
                {
                    "area": item.focus_area,
                    "exercise": item.exercises[0] if item.exercises else "",
                    "time": item.time_required_min
                }
                for item in self.improvement_timeline
                if item.difficulty == "easy" and item.time_required_min <= 10
            ]
        }

    def to_frontend_format(self) -> Dict[str, Any]:
        """Конвертировать в формат для фронтенда"""
        return {
            "overall": self.overall_assessment,
            "strengths": self.strengths,
            "improvements": self.areas_for_improvement,
            "recommendations": self.detailed_recommendations,
            "insights": self.key_insights,
            "confidence": self.confidence_score,

            # Временные данные
            "timeline_analysis": [
                {
                    "time": item.timestamp,
                    "type": item.type,
                    "title": item.title,
                    "description": item.description,
                    "suggestion": item.suggestion,
                    "severity": item.severity,
                    "tags": item.tags
                }
                for item in self.time_based_analysis
            ],

            # Паттерны
            "patterns": [
                {
                    "name": pattern.pattern,
                    "timeRange": pattern.time_range,
                    "description": pattern.description,
                    "recommendation": pattern.recommendation,
                    "impact": pattern.impact
                }
                for pattern in self.temporal_patterns
            ],

            # Критические моменты
            "criticalMoments": [
                {
                    "time": moment.timestamp,
                    "type": moment.type,
                    "description": moment.description,
                    "impact": moment.impact,
                    "lessons": moment.lessons_learned
                }
                for moment in self.critical_moments
            ],

            # План улучшений
            "improvementPlan": {
                "totalTime": sum(item.time_required_min for item in self.improvement_timeline),
                "areas": [
                    {
                        "focus": item.focus_area,
                        "priority": item.priority,
                        "exercises": item.exercises,
                        "expectedImprovement": item.expected_improvement,
                        "timeRequired": item.time_required_min
                    }
                    for item in self.improvement_timeline
                ]
            },

            # Метаданные
            "metadata": {
                "timestamp": self.analysis_timestamp.isoformat(),
                "model": self.model_version,
                "processingTime": self.processing_time_sec
            }
        }
