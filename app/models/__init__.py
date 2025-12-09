# Экспортируем все модели для удобного импорта
from app.models.transcript import Transcript, TranscriptSegment
from app.models.analysis import (
    AnalysisResult,
    FillerWordsStats,
    PausesStats,
    PhraseStats,
    AdviceItem
)
from app.models.gigachat import GigaChatAnalysis

__all__ = [
    "Transcript",
    "TranscriptSegment",
    "AnalysisResult",
    "FillerWordsStats",
    "PausesStats",
    "PhraseStats",
    "AdviceItem",
    "GigaChatAnalysis",
]
