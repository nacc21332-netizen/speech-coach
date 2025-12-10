# Экспортируем все модели для удобного импорта
from .transcript import Transcript, TranscriptSegment, WordTiming
from .analysis import (
    AnalysisResult,
    FillerWordsStats,
    PausesStats,
    PhraseStats,
    AdviceItem
)
from .gigachat import GigaChatAnalysis

__all__ = [
    "Transcript",
    "TranscriptSegment",
    "WordTiming",
    "AnalysisResult",
    "FillerWordsStats",
    "PausesStats",
    "PhraseStats",
    "AdviceItem",
    "GigaChatAnalysis",
]
