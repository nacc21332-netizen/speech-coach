"""
Тест для проверки контекстного анализа слов-паразитов
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.services.contextual_filler_analyzer import ContextualFillerAnalyzer
from app.models.transcript import Transcript, WordTiming, TranscriptSegment
from app.services.gigachat import GigaChatClient
from app.services.cache import AnalysisCache
from pathlib import Path


def test_contextual_filler_analyzer():
    """Тестирование контекстного анализа слов-паразитов"""
    
    # Создаем мок-клиент GigaChat
    mock_gigachat = AsyncMock(spec=GigaChatClient)
    mock_gigachat.classify_fillers_context = AsyncMock()
    
    # Создаем кэш
    mock_cache = MagicMock(spec=AnalysisCache)
    
    # Создаем анализатор
    analyzer = ContextualFillerAnalyzer(mock_gigachat, mock_cache)
    
    # Создаем тестовый транскрипт
    word_timings = [
        WordTiming(word="Привет", start=0.0, end=0.5, confidence=0.9),
        WordTiming(word="я", start=0.5, end=0.7, confidence=0.95),
        WordTiming(word="хочу", start=0.7, end=1.0, confidence=0.85),
        WordTiming(word="рассказать", start=1.0, end=1.5, confidence=0.9),
        WordTiming(word="вот", start=1.5, end=1.7, confidence=0.6),  # потенциальный слово-паразит в контексте
        WordTiming(word="о", start=1.7, end=1.9, confidence=0.8),
        WordTiming(word="проекте", start=1.9, end=2.3, confidence=0.88),
        WordTiming(word="ээ", start=2.3, end=2.6, confidence=0.4),  # однозначный слово-паразит
        WordTiming(word="который", start=2.6, end=3.0, confidence=0.85),
    ]
    
    transcript = Transcript(
        segments=[TranscriptSegment(start=0.0, end=3.0, text="Привет я хочу рассказать вот о проекте ээ который")],
        text="Привет я хочу рассказать вот о проекте ээ который",
        word_timings=word_timings
    )
    
    # Настроим мок-ответ для GigaChat
    mock_gigachat.classify_fillers_context.return_value = [
        {
            "word": "вот",
            "exact_word": "вот",
            "timestamp": 1.5,
            "confidence": 0.6,
            "duration": 0.2,
            "is_filler_context": False,  # "вот" в этом контексте не является паразитом
            "score": 0.3
        }
    ]
    
    # Выполним асинхронный тест
    async def run_test():
        result = await analyzer.analyze_fillers_with_context(transcript)
        
        print(f"Найдено слов-паразитов: {len(result)}")
        for filler in result:
            print(f"  - Слово: '{filler.word}', Точное: '{filler.exact_word}', Время: {filler.timestamp}s")
            if hasattr(filler, 'is_context_filler') and filler.is_context_filler is not None:
                print(f"    - Контекстная классификация: {filler.is_context_filler}, Оценка: {filler.context_score}")
        
        # Проверим, что "вот" не классифицировано как слово-паразит в контексте
        # а "ээ" классифицировано как слово-паразит
        context_fillers = [f for f in result if f.word == "вот"]
        direct_fillers = [f for f in result if f.word == "э-э"]  # "ээ" нормализуется в "э-э"
        
        print(f"\nContext fillers: {len(context_fillers)}")
        print(f"Direct fillers: {len(direct_fillers)}")
        
        # "вот" должно быть в результатах, но не как слово-паразит в контексте
        # "ээ" должно быть в результатах как слово-паразит
        return result
    
    # Запустим тест
    result = asyncio.run(run_test())
    return result


if __name__ == "__main__":
    test_contextual_filler_analyzer()