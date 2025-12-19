from typing import List, Dict, Any, Tuple
import re
from app.models.transcript import Transcript, WordTiming
from app.models.timed_analysis import TimedFillerWord
from app.services.gigachat import GigaChatClient
from app.services.cache import AnalysisCache
from app.core.config import settings


class ContextualFillerAnalyzer:
    """Анализатор слов-паразитов с учетом контекста с помощью GigaChat"""
    
    def __init__(self, gigachat_client: GigaChatClient, cache: AnalysisCache = None):
        self.gigachat_client = gigachat_client
        self.cache = cache
        
        # Слова-паразиты, которые требуют контекстный анализ
        # Words that require contextual analysis (keep conservative core set)
        self.contextual_fillers = {
            "там", "да", "вот", "ну", "как бы", "типа", "значит",
            "вроде", "в общем", "кстати", "собственно", "то есть", "наверное", "кажется",
            "по сути", "всё-таки", "прямо"
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
            (r"\bнаверное\b", "наверное"),
            (r"\bкажется\b", "кажется"),
            (r"\bпо сути\b", "по сути"),
            (r"\bвсё-таки\b", "всё-таки"),
            (r"\bпрямо\b", "прямо"),
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

    def _find_candidate_fillers(self, transcript: Transcript) -> List[Dict[str, Any]]:
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

    async def analyze_fillers_with_context(self, transcript: Transcript) -> List[TimedFillerWord]:
        """Анализирует слова-паразиты с учетом контекста с помощью GigaChat"""
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
                direct_fillers.append(TimedFillerWord(
                    word=candidate["word"],
                    timestamp=candidate["timestamp"],
                    exact_word=candidate["exact_word"],
                    confidence=candidate["confidence"],
                    duration=candidate["duration"]
                ))
        
        # Для контекстных слов используем GigaChat
        if contextual_candidates and settings.llm_fillers_enabled:
            try:
                classified_candidates = await self.gigachat_client.classify_fillers_context(
                    contextual_candidates, 
                    self.cache
                )
                
                # Фильтруем только те, которые действительно являются словами-паразитами в контексте
                contextual_fillers = []
                for candidate in classified_candidates:
                    # Создаем объект TimedFillerWord с контекстной информацией
                    timed_filler = TimedFillerWord(
                        word=candidate["word"],
                        timestamp=candidate["timestamp"],
                        exact_word=candidate["exact_word"],
                        confidence=candidate["confidence"],
                        duration=candidate["duration"],
                        is_context_filler=candidate.get("is_filler", False),
                        context_score=candidate.get("confidence", 0.0)
                    )
                    contextual_fillers.append(timed_filler)
                
                # Объединяем результаты
                return direct_fillers + contextual_fillers
            except Exception as e:
                # В случае ошибки возвращаем только прямые совпадения
                return direct_fillers
        else:
            # Если GigaChat отключен, возвращаем все кандидаты как слова-паразиты
            return [TimedFillerWord(
                word=candidate["word"],
                timestamp=candidate["timestamp"],
                exact_word=candidate["exact_word"],
                confidence=candidate["confidence"],
                duration=candidate["duration"]
            ) for candidate in candidates]

    def _find_fillers_with_exact_timings(self, transcript: Transcript) -> List[TimedFillerWord]:
        """Устаревший метод, оставлен для совместимости"""
        # Этот метод будет асинхронным в новой реализации
        return []