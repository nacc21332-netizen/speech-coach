"""
Модели для анализа с временными метками.
"""
from pydantic import BaseModel
from typing import List, Dict, Literal, Optional, Any


class TimedFillerWord(BaseModel):
    """Слово-паразит с временной меткой"""
    word: str
    timestamp: float
    text_context: str
    segment_start: float
    segment_end: float


class TimedPause(BaseModel):
    """Пауза с контекстом"""
    start: float
    end: float
    duration: float
    type: Literal["natural", "long", "awkward", "dramatic"]
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    speech_before_end: float
    speech_after_start: float


class SpeechRateWindow(BaseModel):
    """Темп речи в временном окне"""
    window_start: float
    window_end: float
    word_count: int
    words_per_minute: float
    speaking_time: float


class EmotionalPeak(BaseModel):
    """Эмоциональный пик"""
    timestamp: float
    intensity: float  # 0-1
    type: Literal["volume", "speed", "pause"]
    description: str


class AdviceItem(BaseModel):
    """Совет по улучшению речи"""
    category: Literal["speech_rate", "filler_words", "pauses", "phrasing"]
    severity: Literal["info", "suggestion", "warning"]
    title: str
    observation: str
    recommendation: str


class FillerWordsStats(BaseModel):
    """Статистика по словам-паразитам"""
    total: int
    per_100_words: float
    items: List[Dict[str, Any]]


class PausesStats(BaseModel):
    """Статистика по паузам"""
    count: int
    avg_sec: float
    max_sec: float
    long_pauses: List[Dict[str, float]]


class PhraseStats(BaseModel):
    """Статистика по фразам"""
    count: int
    avg_words: float
    avg_duration_sec: float
    min_words: int
    max_words: int
    min_duration_sec: float
    max_duration_sec: float
    length_classification: str
    rhythm_variation: str


class TimedAnalysisResult(BaseModel):
    """Результат анализа с временными метками"""
    # Основные метрики
    duration_sec: float
    speaking_time_sec: float
    speaking_ratio: float
    words_total: int
    words_per_minute: float

    # Статистика
    filler_words: FillerWordsStats
    pauses: PausesStats
    phrases: PhraseStats

    # Советы
    advice: List[AdviceItem]

    # Детализированные данные с таймингами
    filler_words_detailed: List[TimedFillerWord]
    pauses_detailed: List[TimedPause]
    speech_rate_windows: List[SpeechRateWindow]
    emotional_peaks: List[EmotionalPeak]
    phrase_boundaries: List[float]

    # Для визуализации
    speaking_activity: List[Dict[str, float]]

    # Исходные данные
    transcript_segments: List[Dict[str, Any]]
    transcript_text: str
