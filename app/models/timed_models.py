"""
Модели для детализированных таймингов речи.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional, Any
from enum import Enum


class SpeechElementType(str, Enum):
    """Типы элементов речи"""
    WORD = "word"
    FILLER = "filler"
    PAUSE = "pause"
    PHRASE = "phrase"
    QUESTION = "question"
    EMPHASIS = "emphasis"
    HESITATION = "hesitation"
    PROBLEMATIC = "problematic"


class WordTiming(BaseModel):
    """Тайминг для отдельного слова с расширенной информацией"""
    word: str = Field(description="Текст слова")
    start: float = Field(description="Время начала в секундах")
    end: float = Field(description="Время окончания в секундах")
    duration: float = Field(description="Длительность слова в секундах")
    confidence: Optional[float] = Field(
        None, description="Уверенность распознавания (0-1)")
    type: SpeechElementType = Field(
        SpeechElementType.WORD, description="Тип элемента")
    is_filler: bool = Field(False, description="Является ли словом-паразитом")
    is_hesitation: bool = Field(False, description="Является ли колебанием")
    context_before: Optional[str] = Field(
        None, description="Контекст перед словом (2-3 слова)")
    context_after: Optional[str] = Field(
        None, description="Контекст после слова (2-3 слова)")
    emotion_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка эмоциональности (0-1)")


class FillerWordDetail(BaseModel):
    """Детализированное слово-паразит"""
    word: str = Field(
        description="Нормализованное слово-паразит (например, 'э-э')")
    timestamp: float = Field(description="Время появления в секундах")
    exact_word: str = Field(description="Точное распознанное слово")
    duration: float = Field(description="Длительность произнесения в секундах")
    confidence: Optional[float] = Field(
        None, description="Уверенность распознавания")
    frequency: int = Field(1, description="Частота появления в выступлении")
    context_window: str = Field("", description="Окно контекста (±2 слова)")
    segment_index: int = Field(
        0, description="Индекс в последовательности слов")
    severity: Literal["low", "medium", "high"] = Field(
        "low", description="Серьезность проблемы")
    suggestions: List[str] = Field(
        default_factory=list, description="Рекомендации по исправлению")
    # LLM contextual classification
    is_context_filler: Optional[bool] = Field(None, description="Определено ли слово как паразит в контексте LLM")
    context_score: Optional[float] = Field(None, description="Оценка уверенности LLM (0-1)")


class PauseDetail(BaseModel):
    """Детализированная пауза"""
    start: float = Field(description="Начало паузы в секундах")
    end: float = Field(description="Конец паузы в секундах")
    duration: float = Field(description="Длительность паузы в секундах")
    type: Literal["natural", "long", "awkward", "dramatic",
                  "hesitation"] = Field("natural", description="Тип паузы")
    intensity: float = Field(
        0.5, ge=0.0, le=1.0, description="Интенсивность паузы (0-1)")
    context_before: str = Field("", description="Контекст перед паузой")
    context_after: str = Field("", description="Контекст после паузы")
    before_word: Optional[str] = Field(None, description="Слово перед паузой")
    after_word: Optional[str] = Field(None, description="Слово после паузой")
    is_excessive: bool = Field(
        False, description="Является ли пауза чрезмерно длинной")
    natural_threshold: float = Field(
        2.5, description="Порог естественной паузы в секундах")
    recommendation: Optional[str] = Field(
        None, description="Рекомендация по паузе")


class PhraseDetail(BaseModel):
    """Детализированная фраза"""
    id: int = Field(description="Уникальный идентификатор фразы")
    start: float = Field(description="Начало фразы в секундах")
    end: float = Field(description="Конец фразы в секундах")
    duration: float = Field(description="Длительность фразы в секундах")
    word_count: int = Field(description="Количество слов во фразе")
    words_per_minute: float = Field(description="Темп речи в словах в минуту")
    text: str = Field(description="Текст фразы")
    words: List[WordTiming] = Field(
        default_factory=list, description="Слова в фразе с таймингами")
    filler_count: int = Field(
        0, description="Количество слов-паразитов во фразе")
    pause_count: int = Field(0, description="Количество пауз внутри фразы")
    avg_pause_duration: float = Field(
        0.0, description="Средняя длительность пауз в фразе")
    complexity_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Сложность фразы (0-1)")
    is_complex: bool = Field(False, description="Является ли фраза сложной")
    clarity_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Оценка ясности (0-1)")
    emotion_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Оценка эмоциональности (0-1)")


class QuestionDetail(BaseModel):
    """Вопросы в речи"""
    text: str = Field(description="Текст вопроса")
    timestamp: float = Field(description="Время начала вопроса в секундах")
    duration: float = Field(description="Длительность вопроса в секундах")
    word_count: int = Field(description="Количество слов в вопросе")
    is_rhetorical: bool = Field(
        False, description="Является ли вопрос риторическим")
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Уверенность в классификации")
    answer_prompted: bool = Field(
        False, description="Требует ли вопрос ответа от аудитории")
    effectiveness_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Эффективность вопроса")


class EmphasisDetail(BaseModel):
    """Акценты/эмоциональные моменты"""
    timestamp: float = Field(description="Время акцента в секундах")
    type: Literal["volume", "speed", "pitch", "pause",
                  "repetition", "duration", "content"] = Field("volume", description="Тип акцента")
    intensity: float = Field(
        0.5, ge=0.0, le=1.0, description="Интенсивность акцента (0-1)")
    description: str = Field(description="Описание акцента")
    context: str = Field("", description="Контекст акцента")
    effectiveness: float = Field(
        0.0, ge=0.0, le=1.0, description="Эффективность акцента")
    suggestion: Optional[str] = Field(
        None, description="Рекомендация по акценту")


class SuspiciousMoment(BaseModel):
    """Сомнительные/проблемные моменты"""
    id: int = Field(description="Уникальный идентификатор момента")
    timestamp: float = Field(description="Время момента в секундах")
    type: Literal["long_pause", "excessive_filler", "fast_speech",
                  "slow_speech", "repetition", "hesitation",
                  "incoherence", "monotone", "volume_drop", 
                  "stuttering", "trailing_off", "excessive_emphasis",
                  "frequent_correction"] = Field("long_pause", description="Тип проблемы")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        "medium", description="Серьезность проблемы")
    duration: float = Field(description="Длительность проблемы в секундах")
    description: str = Field(description="Подробное описание проблемы")
    suggestion: str = Field(
        description="Конкретная рекомендация по исправлению")
    context_before: str = Field("", description="Контекст перед проблемой")
    context_after: str = Field("", description="Контекст после проблемы")
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Уверенность в определении проблемы")
    words_affected: List[str] = Field(
        default_factory=list, description="Слова, затронутые проблемой")
    improvement_potential: float = Field(
        0.0, ge=0.0, le=1.0, description="Потенциал улучшения (0-1)")


class SpeechTimeline(BaseModel):
    """Временная шкала всех элементов речи"""
    words: List[WordTiming] = Field(
        default_factory=list, description="Все слова с таймингами")
    fillers: List[FillerWordDetail] = Field(
        default_factory=list, description="Слова-паразиты")
    pauses: List[PauseDetail] = Field(
        default_factory=list, description="Паузы")
    phrases: List[PhraseDetail] = Field(
        default_factory=list, description="Фразы")
    questions: List[QuestionDetail] = Field(
        default_factory=list, description="Вопросы")
    emphases: List[EmphasisDetail] = Field(
        default_factory=list, description="Акценты")
    suspicious_moments: List[SuspiciousMoment] = Field(
        default_factory=list, description="Сомнительные моменты")

    # Методы для удобства
    def get_moment_at_time(self, time: float, tolerance: float = 0.5) -> Optional[Dict]:
        """Получить элемент речи в указанное время"""
        for element_type in ["words", "fillers", "pauses", "questions", "emphases", "suspicious_moments"]:
            elements = getattr(self, element_type)
            for element in elements:
                if hasattr(element, 'start') and hasattr(element, 'end'):
                    if element.start - tolerance <= time <= element.end + tolerance:
                        return {
                            "type": element_type[:-1],  # Убираем 's' в конце
                            "element": element,
                            "exact_match": element.start <= time <= element.end
                        }
        return None

    def get_words_in_range(self, start: float, end: float) -> List[WordTiming]:
        """Получить слова в указанном временном диапазоне"""
        return [w for w in self.words if start <= w.start <= end]

    def get_problems_in_range(self, start: float, end: float) -> List[SuspiciousMoment]:
        """Получить проблемные моменты в указанном диапазоне"""
        return [p for p in self.suspicious_moments if start <= p.timestamp <= end]


class TimedAnalysisResult(BaseModel):
    """Полный результат анализа с таймингами"""
    # Основные метрики
    duration_sec: float = Field(description="Общая длительность в секундах")
    speaking_time_sec: float = Field(description="Время говорения в секундах")
    speaking_ratio: float = Field(description="Коэффициент говорения (0-1)")
    words_total: int = Field(description="Общее количество слов")
    words_per_minute: float = Field(
        description="Средний темп речи (слов/минуту)")

    # Статистика
    filler_words: Dict[str, Any] = Field(
        description="Статистика по словам-паразитам")
    pauses: Dict[str, Any] = Field(description="Статистика по паузам")
    phrases: Dict[str, Any] = Field(description="Статистика по фразам")

    # Советы
    advice: List[Dict[str, Any]] = Field(
        default_factory=list, description="Общие рекомендации")

    # Полный транскрипт
    transcript: str = Field(description="Полный текст транскрипции")

    # Детализированные тайминги
    timeline: SpeechTimeline = Field(
        default_factory=SpeechTimeline, description="Временная шкала элементов речи")

    # Визуализация
    speech_activity: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Данные для графика активности речи"
    )
    speech_rate_windows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Темп речи по временным окнам"
    )
    intensity_profile: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Профиль интенсивности речи"
    )
    emotion_timeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Временная шкала эмоциональности"
    )

    # Анализ GigaChat
    gigachat_analysis: Optional[Any] = Field(
        None, description="Расширенный анализ от GigaChat API")

    # Метаданные
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "analysis_version": "2.0",
            "includes_timings": True,
            "includes_visualization": True,
            "timestamp": None,
            "processing_time_sec": None
        },
        description="Метаданные анализа"
    )

    # Методы для удобства
    def get_summary(self) -> Dict[str, Any]:
        """Получить краткое резюме анализа"""
        return {
            "duration": f"{self.duration_sec:.1f} сек",
            "speaking_time": f"{self.speaking_time_sec:.1f} сек",
            "speaking_ratio": f"{self.speaking_ratio:.1%}",
            "words": self.words_total,
            "words_per_minute": f"{self.words_per_minute:.1f}",
            "filler_words": self.filler_words.get("total", 0),
            "pauses": self.pauses.get("total", 0),
            "phrases": self.phrases.get("total", 0),
            "problems": len(self.timeline.suspicious_moments),
            "has_gigachat": self.gigachat_analysis is not None
        }

    def get_problem_areas(self) -> List[Dict[str, Any]]:
        """Получить основные проблемные области"""
        problems_by_type = {}
        for moment in self.timeline.suspicious_moments:
            if moment.type not in problems_by_type:
                problems_by_type[moment.type] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "max_severity": "low",
                    "examples": []
                }

            problems_by_type[moment.type]["count"] += 1
            problems_by_type[moment.type]["total_duration"] += moment.duration

            # Обновляем максимальную серьезность
            severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            current_max = problems_by_type[moment.type]["max_severity"]
            if severity_order[moment.severity] > severity_order[current_max]:
                problems_by_type[moment.type]["max_severity"] = moment.severity

            # Добавляем пример (первые 3)
            if len(problems_by_type[moment.type]["examples"]) < 3:
                problems_by_type[moment.type]["examples"].append({
                    "time": f"{moment.timestamp:.1f}с",
                    "description": moment.description[:100],
                    "suggestion": moment.suggestion[:100]
                })

        return [
            {
                "type": problem_type,
                **details
            }
            for problem_type, details in problems_by_type.items()
        ]

    def get_timeline_for_frontend(self) -> Dict[str, Any]:
        """Подготовить данные временной шкалы для фронтенда"""
        return {
            "words": [
                {
                    "text": w.word,
                    "start": w.start,
                    "end": w.end,
                    "is_filler": w.is_filler,
                    "is_hesitation": w.is_hesitation,
                    "confidence": w.confidence
                }
                # Ограничиваем для производительности
                for w in self.timeline.words[:500]
            ],
            "fillers": [
                {
                    "word": f.word,
                    "time": f.timestamp,
                    "duration": f.duration,
                    "severity": f.severity,
                    "context": f.context_window
                }
                for f in self.timeline.fillers
            ],
            "pauses": [
                {
                    "start": p.start,
                    "end": p.end,
                    "duration": p.duration,
                    "type": p.type,
                    "is_excessive": p.is_excessive,
                    "context": f"{p.context_before} [...] {p.context_after}"
                }
                # Только заметные паузы
                for p in self.timeline.pauses if p.duration > 1.0
            ],
            "problems": [
                {
                    "id": p.id,
                    "time": p.timestamp,
                    "type": p.type,
                    "severity": p.severity,
                    "description": p.description,
                    "suggestion": p.suggestion,
                    "context": f"{p.context_before} [...] {p.context_after}"
                }
                for p in self.timeline.suspicious_moments
            ],
            "phrases": [
                {
                    "id": p.id,
                    "start": p.start,
                    "end": p.end,
                    "text": p.text[:100] + ("..." if len(p.text) > 100 else ""),
                    "word_count": p.word_count,
                    "wpm": p.words_per_minute,
                    "filler_count": p.filler_count,
                    "is_complex": p.is_complex
                }
                for p in self.timeline.phrases
            ]
        }
