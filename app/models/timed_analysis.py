"""
Модели для анализа с временными метками.
"""
from pydantic import BaseModel
from typing import List, Dict, Literal, Optional, Any


from app.models.timed_models import FillerWordDetail as TimedFillerWord


from app.models.timed_models import PauseDetail as TimedPause


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
    type: Literal["volume", "speed", "pause", "duration", "repetition", "content"]
    description: str


class AdviceItem(BaseModel):
    """Совет по улучшению речи"""
    category: Literal["speech_rate", "filler_words", "pauses", "phrasing"]
    severity: Literal["info", "suggestion", "warning"]
    title: str
    observation: str
    recommendation: str


from app.models.analysis import FillerWordsStats, PausesStats, PhraseStats


from app.models.timed_models import TimedAnalysisResult
# Re-export TimedAnalysisResult from `timed_models.py` to avoid duplication and maintain compatibility
TimedAnalysisResult = TimedAnalysisResult


class TimedAnalysisData(BaseModel):
    """Дополнительные данные с таймингами"""
    filler_words_detailed: List[TimedFillerWord] = []
    pauses_detailed: List[TimedPause] = []
    speech_rate_windows: List[SpeechRateWindow] = []
    word_timings_count: int = 0
    speaking_activity: List[Dict[str, float]] = []
