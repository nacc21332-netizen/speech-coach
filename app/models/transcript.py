"""
Модели для транскрипта с таймингами слов.
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class WordTiming(BaseModel):
    """Тайминг для отдельного слова"""
    word: str
    start: float
    end: float
    confidence: Optional[float] = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: List[WordTiming] = []  # ← НОВОЕ: тайминги слов внутри сегмента


class Transcript(BaseModel):
    text: str
    segments: List[TranscriptSegment]
    word_timings: List[WordTiming] = []  # Все слова с таймингами
