"""
Простой тест для проверки логики контекстного анализа слов-паразитов
"""
import re
from typing import List, Dict, Any, Optional


class MockWordTiming:
    def __init__(self, word: str, start: float, end: float, confidence: Optional[float] = None):
        self.word = word
        self.start = start
        self.end = end
        self.confidence = confidence


class MockTranscript:
    def __init__(self, word_timings: List[MockWordTiming], segments=None, text=""):
        self.word_timings = word_timings
        self.segments = segments or []
        self.text = text


class MockTimedFillerWord:
    def __init__(self, word: str, timestamp: float, exact_word: str, duration: float, confidence: Optional[float] = None):
        self.word = word
        self.timestamp = timestamp
        self.exact_word = exact_word
        self.duration = duration
        self.confidence = confidence
        # Добавим поля для контекстной классификации
        self.is_context_filler = None
        self.context_score = None


class SimpleContextualFillerAnalyzer:
    """Упрощенная версия анализатора для тестирования логики"""
    
    def __init__(self):
        # Слова-паразиты, которые требуют контекстный анализ
        self.contextual_fillers = {
            "там", "да", "вот", "ну", "как бы", "типа", "значит", 
            "вроде", "в общем", "кстати", "собственно", "то есть"
        }
        
        # Регулярные выражения для слов-паразитов
        self.filler_patterns = [
            (r"\bэ+([- ]э+)*\b", "э-э"),
            (r"\bэм+\b", "эм"),
            (r"\bмм+\b", "мм"),
            (r"\bну\b", "ну"),
            (r"\bвот\b", "вот"),
            (r"\bтам\b", "там"),
            (r"\bда\b", "да"),
            (r"\bкороче\b", "короче"),
            (r"\bвроде\b", "вроде"),
            (r"\bвроде бы\b", "вроде бы"),
            (r"\bв общем\b", "в общем"),
            (r"\bкстати\b", "кстати"),
            (r"\bсобственно\b", "собственно"),
            (r"\bкак бы\b", "как бы"),
            (r"\bтипа\b", "типа"),
            (r"\bто есть\b", "то есть"),
            (r"\bзначит\b", "значит"),
            (r"\bполучается\b", "получается"),
        ]

    def _find_candidate_fillers(self, transcript: MockTranscript) -> List[Dict[str, Any]]:
        """Находит кандидатов на слова-паразиты без учета контекста"""
        candidates = []
        
        for i, word_timing in enumerate(transcript.word_timings):
            word_text = word_timing.word.lower().strip().strip(",.!?;:()\"'")
            word_text_norm = re.sub(r"(.)\1{2,}", r"\1\1", word_text)
            word_text_norm = word_text_norm.replace("-", " ")
            
            for pattern, filler_name in self.filler_patterns:
                compiled_pattern = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)
                
                if compiled_pattern.search(word_text) or compiled_pattern.search(word_text_norm) or (filler_name in word_text_norm):
                    # Получаем контекст: предыдущие и следующие слова
                    context_before = ""
                    context_after = ""
                    
                    # Берем 3 слова до и 3 слова после
                    start_idx = max(0, i - 3)
                    end_idx = min(len(transcript.word_timings), i + 4)
                    
                    for j in range(start_idx, i):
                        context_before += transcript.word_timings[j].word + " "
                    
                    for j in range(i + 1, end_idx):
                        context_after += transcript.word_timings[j].word + " "
                    
                    context_before = context_before.strip()
                    context_after = context_after.strip()
                    
                    candidates.append({
                        "word": filler_name,
                        "exact_word": word_timing.word,
                        "timestamp": round(word_timing.start, 3),
                        "confidence": word_timing.confidence,
                        "duration": round(max(0.0, word_timing.end - word_timing.start), 3),
                        "word_timing": word_timing,
                        "context_before": context_before,
                        "context_after": context_after
                    })
                    break
        
        return candidates

    def analyze_fillers_with_context(self, transcript: MockTranscript) -> List[MockTimedFillerWord]:
        """Симуляция анализа слов-паразитов с учетом контекста"""
        # Сначала находим все кандидаты
        candidates = self._find_candidate_fillers(transcript)
        
        if not candidates:
            return []
        
        # Фильтруем только те, которые требуют контекстный анализ
        contextual_candidates = []
        for candidate in candidates:
            word_lower = candidate["word"].lower()
            if word_lower in self.contextual_fillers:
                contextual_candidates.append(candidate)
            else:
                # Для однозначных слов-паразитов (например, "э-э", "эм") сразу добавляем в результат
                pass
        
        # Для не-контекстных слов-паразитов сразу создаем результат
        direct_fillers = []
        for candidate in candidates:
            word_lower = candidate["word"].lower()
            if word_lower not in self.contextual_fillers:
                filler = MockTimedFillerWord(
                    word=candidate["word"],
                    timestamp=candidate["timestamp"],
                    exact_word=candidate["exact_word"],
                    duration=candidate["duration"],
                    confidence=candidate["confidence"]
                )
                # Эти слова всегда считаются словами-паразитами
                filler.is_context_filler = True
                filler.context_score = 1.0
                direct_fillers.append(filler)
        
        # Для контекстных слов симулируем анализ
        contextual_fillers = []
        for candidate in candidates:
            word_lower = candidate["word"].lower()
            if word_lower in self.contextual_fillers:
                filler = MockTimedFillerWord(
                    word=candidate["word"],
                    timestamp=candidate["timestamp"],
                    exact_word=candidate["exact_word"],
                    duration=candidate["duration"],
                    confidence=candidate["confidence"]
                )
                
                # Симуляция контекстного анализа:
                # "вот" в начале фразы или как указательное местоимение не является словом-паразитом
                # "вот" как заполнитель паузы - является словом-паразитом
                if word_lower == "вот":
                    # Проверим контекст: если после "вот" идет существительное, это может быть нормальным использованием
                    context_after = candidate["context_after"].lower().split()
                    if context_after and context_after[0] in ["проекте", "деле", "теме", "вопросе", "случае"]:
                        # В этих контекстах "вот" используется как указательное местоимение, не слово-паразит
                        filler.is_context_filler = False
                        filler.context_score = 0.2
                    else:
                        # В других контекстах может быть словом-паразитом
                        filler.is_context_filler = True
                        filler.context_score = 0.7
                else:
                    # Для других потенциально контекстных слов-паразитов
                    filler.is_context_filler = True
                    filler.context_score = 0.6
                
                contextual_fillers.append(filler)
        
        # Объединяем результаты
        return direct_fillers + contextual_fillers


def test_simple_contextual_analyzer():
    """Тестирование упрощенного контекстного анализатора"""
    analyzer = SimpleContextualFillerAnalyzer()
    
    # Создаем тестовый транскрипт
    word_timings = [
        MockWordTiming("Привет", 0.0, 0.5, 0.9),
        MockWordTiming("я", 0.5, 0.7, 0.95),
        MockWordTiming("хочу", 0.7, 1.0, 0.85),
        MockWordTiming("рассказать", 1.0, 1.5, 0.9),
        MockWordTiming("вот", 1.5, 1.7, 0.6),  # потенциальный слово-паразит в контексте "вот о проекте"
        MockWordTiming("о", 1.7, 1.9, 0.8),
        MockWordTiming("проекте", 1.9, 2.3, 0.88),
        MockWordTiming("ээ", 2.3, 2.6, 0.4),  # однозначный слово-паразит
        MockWordTiming("который", 2.6, 3.0, 0.85),
        MockWordTiming("вот", 3.0, 3.2, 0.7),  # другой случай "вот" - возможно слово-паразит
        MockWordTiming("реализуем", 3.2, 3.7, 0.85),
    ]
    
    transcript = MockTranscript(word_timings, text="Привет я хочу рассказать вот о проекте ээ который вот реализуем")
    
    result = analyzer.analyze_fillers_with_context(transcript)
    
    print(f"Найдено слов-паразитов: {len(result)}")
    for filler in result:
        print(f"  - Слово: '{filler.word}', Точное: '{filler.exact_word}', Время: {filler.timestamp}s")
        if hasattr(filler, 'is_context_filler') and filler.is_context_filler is not None:
            status = "паразит" if filler.is_context_filler else "не паразит"
            print(f"    - Контекстная классификация: {status}, Оценка: {filler.context_score}")
            print(f"    - Контекст: '{analyzer._find_candidate_fillers(transcript)[result.index(filler)]['context_before']} | {analyzer._find_candidate_fillers(transcript)[result.index(filler)]['context_after']}'")
    
    # Подсчет результатов
    context_fillers = [f for f in result if f.is_context_filler is False]  # Не являются словами-паразитами в контексте
    actual_fillers = [f for f in result if f.is_context_filler is True]    # Являются словами-паразитами
    
    print(f"\nСлов, НЕ являющихся паразитами в контексте: {len(context_fillers)}")
    print(f"Слов, являющихся паразитами: {len(actual_fillers)}")
    
    return result


if __name__ == "__main__":
    test_simple_contextual_analyzer()