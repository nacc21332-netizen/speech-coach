"""
Продвинутый пайплайн с поддержкой детализированных таймингов.
"""
import logging
import asyncio
from typing import Optional
from pathlib import Path
import json

from fastapi import UploadFile

from app.services.transcriber import LocalWhisperTranscriber
from app.services.analyzer_advanced import AdvancedSpeechAnalyzer
from app.services.gigachat import GigaChatClient
from app.services.pipeline import SpeechAnalysisPipeline as BasePipeline
from app.models.timed_models import TimedAnalysisResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class AdvancedSpeechAnalysisPipeline(BasePipeline):
    """Продвинутый пайплайн с детализированными таймингами"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.advanced_analyzer = AdvancedSpeechAnalyzer()

    async def analyze_with_timings(self, file: UploadFile) -> TimedAnalysisResult:
        """
        Анализирует файл с полными таймингами.
        """
        # Используем базовый пайплайн для извлечения и транскрипции
        temp_video_path, temp_audio_path = await self._create_temp_files(file)

        try:
            # 1. Извлечение аудио
            await self._extract_audio(temp_video_path, temp_audio_path)

            # 2. Транскрипция с таймингами
            transcript = await self._transcribe_audio(temp_audio_path)

            # 3. Продвинутый анализ с таймингами (передаем путь к аудио для RMS-показателей)
            result = await self.advanced_analyzer.analyze_with_timings(transcript, temp_audio_path)

            # 4. GigaChat анализ (если включен)
            if self.gigachat_client:
                result = await self._enhance_with_gigachat_advanced(result)

            # Используем LLM для классификации слов-паразитов по контексту (если включено)
            if self.gigachat_client and settings.llm_fillers_enabled and result.timeline.fillers:
                try:
                    contexts = []
                    for filler in result.timeline.fillers:
                        # find index of word timing closest to filler.timestamp
                        nearest_idx = None
                        min_diff = float('inf')
                        for i, wt in enumerate(transcript.word_timings):
                            diff = abs(wt.start - filler.timestamp)
                            if diff < min_diff:
                                min_diff = diff
                                nearest_idx = i
                        before_words = []
                        after_words = []
                        if nearest_idx is not None:
                            for j in range(max(0, nearest_idx - 2), nearest_idx):
                                before_words.append(transcript.word_timings[j].word)
                            for j in range(nearest_idx + 1, min(len(transcript.word_timings), nearest_idx + 3)):
                                after_words.append(transcript.word_timings[j].word)
                        contexts.append({
                            "word": filler.word,
                            "exact_word": filler.exact_word,
                            "timestamp": filler.timestamp,
                            "context_before": " ".join(before_words),
                            "context_after": " ".join(after_words)
                        })
                    classified = await self.gigachat_client.classify_fillers_context(contexts, cache=self.cache)
                    if classified:
                        for idx, cl in enumerate(classified):
                            if idx < len(result.timeline.fillers):
                                result.timeline.fillers[idx].is_context_filler = cl.get('is_filler', cl.get('is_filler_context', False))
                                result.timeline.fillers[idx].context_score = cl.get('score', cl.get('confidence', cl.get('confidence_score')))
                except Exception as e:
                    logger.warning(f"LLM filler classification (advanced) failed: {e}")

            return result

        finally:
            self._cleanup_temp_files(temp_video_path, temp_audio_path)

    async def _enhance_with_gigachat_advanced(self, result: TimedAnalysisResult) -> TimedAnalysisResult:
        """Расширенный анализ через GigaChat с учетом таймингов"""
        if not self.gigachat_client:
            return result

        try:
            # Создаем промпт с детализированными таймингами
            prompt = self._create_detailed_prompt(result)

            # Используем существующий метод analyze_speech, но передаем ему словарь
            # вместо объекта AnalysisResult
            gigachat_input = {
                "duration_sec": result.duration_sec,
                "speaking_time_sec": result.speaking_time_sec,
                "speaking_ratio": result.speaking_ratio,
                "words_total": result.words_total,
                "words_per_minute": result.words_per_minute,
                "transcript": result.transcript[:3000],  # Ограничиваем длину
                "timeline": {
                    "fillers": [
                        {
                            "word": f.word,
                            "timestamp": f.timestamp,
                            "exact_word": f.exact_word,
                            "duration": f.duration,
                            "severity": f.severity
                        }
                        # Ограничиваем количество
                        for f in result.timeline.fillers[:50]
                    ],
                    "pauses": [
                        {
                            "start": p.start,
                            "end": p.end,
                            "duration": p.duration,
                            "type": p.type,
                            "is_excessive": p.is_excessive
                        }
                        for p in result.timeline.pauses if p.duration > 1.0
                    ][:30],
                    "suspicious_moments": [
                        {
                            "timestamp": m.timestamp,
                            "type": m.type,
                            "severity": m.severity,
                            "description": m.description,
                            "suggestion": m.suggestion
                        }
                        for m in result.timeline.suspicious_moments[:20]
                    ]
                }
            }

            # Используем метод analyze_speech_with_timings если он есть,
            # иначе используем обычный analyze_speech
            if hasattr(self.gigachat_client, 'analyze_speech_with_timings'):
                gigachat_analysis = await self.gigachat_client.analyze_speech_with_timings(gigachat_input)
            else:
                # Создаем упрощенный AnalysisResult для обратной совместимости
                from app.models.analysis import AnalysisResult, FillerWordsStats, PausesStats, PhraseStats

                base_result = AnalysisResult(
                    duration_sec=result.duration_sec,
                    speaking_time_sec=result.speaking_time_sec,
                    speaking_ratio=result.speaking_ratio,
                    words_total=result.words_total,
                    words_per_minute=result.words_per_minute,
                    filler_words=FillerWordsStats(
                        total=len(result.timeline.fillers),
                        per_100_words=len(
                            result.timeline.fillers) / result.words_total * 100 if result.words_total > 0 else 0,
                        items=[{"word": f.word, "count": 1}
                               for f in result.timeline.fillers[:10]]
                    ),
                    pauses=PausesStats(
                        count=len(result.timeline.pauses),
                        avg_sec=sum(p.duration for p in result.timeline.pauses) /
                        len(result.timeline.pauses) if result.timeline.pauses else 0,
                        max_sec=max(
                            (p.duration for p in result.timeline.pauses), default=0),
                        long_pauses=[
                            {"start": p.start, "end": p.end, "duration": p.duration}
                            for p in result.timeline.pauses if p.duration > 2.5
                        ][:3]
                    ),
                    phrases=PhraseStats(
                        count=len(result.timeline.phrases),
                        avg_words=sum(p.word_count for p in result.timeline.phrases) / len(
                            result.timeline.phrases) if result.timeline.phrases else 0,
                        avg_duration_sec=sum(p.duration for p in result.timeline.phrases) / len(
                            result.timeline.phrases) if result.timeline.phrases else 0,
                        min_words=min(
                            (p.word_count for p in result.timeline.phrases), default=0),
                        max_words=max(
                            (p.word_count for p in result.timeline.phrases), default=0),
                        min_duration_sec=min(
                            (p.duration for p in result.timeline.phrases), default=0),
                        max_duration_sec=max(
                            (p.duration for p in result.timeline.phrases), default=0),
                        length_classification="balanced",
                        rhythm_variation="moderately_variable"
                    ),
                    advice=result.advice,
                    transcript=result.transcript,
                    gigachat_analysis=None
                )

                gigachat_analysis = await self.gigachat_client.analyze_speech(base_result)

            if gigachat_analysis:
                result.gigachat_analysis = gigachat_analysis
                logger.info("GigaChat анализ с таймингами получен")

        except Exception as e:
            logger.error(f"Ошибка GigaChat анализа: {e}")

        return result

    def _create_detailed_prompt(self, result: TimedAnalysisResult) -> str:
        """Создает детализированный промпт для GigaChat"""
        # Основные проблемы
        problems_summary = []
        for moment in result.timeline.suspicious_moments[:10]:
            problems_summary.append(
                f"- {moment.timestamp:.1f}с: {moment.description} "
                f"(серьезность: {moment.severity})"
            )

        # Слова-паразиты
        filler_summary = []
        for filler in result.timeline.fillers[:15]:
            filler_summary.append(
                f"- {filler.timestamp:.1f}с: '{filler.exact_word}' "
                f"(серьезность: {filler.severity})"
            )

        # Паузы
        long_pauses = [p for p in result.timeline.pauses if p.is_excessive]
        pause_summary = []
        for pause in long_pauses[:10]:
            pause_summary.append(
                f"- {pause.start:.1f}с: {pause.duration:.1f}сек "
                f"({pause.type})"
            )

        # Вопросы
        question_summary = []
        for question in result.timeline.questions[:5]:
            q_type = "риторический" if question.is_rhetorical else "обычный"
            question_summary.append(
                f"- {question.timestamp:.1f}с: '{question.text[:50]}...' ({q_type})"
            )

        # Акценты
        emphasis_summary = []
        for emphasis in result.timeline.emphases[:5]:
            emphasis_summary.append(
                f"- {emphasis.timestamp:.1f}с: {emphasis.type} - '{emphasis.description}'"
            )

        # Produce a concise, structured prompt that asks for time-aligned, actionable
        # recommendations. Ask the LLM to be conservative and avoid contradictory advice
        # (e.g. don't both ask to keep and remove pauses without context).
        prompt = f"""
    Пожалуйста, проанализируй выступление и верни строго корректный JSON-объект.

    Контекст (основные метрики):
    duration_sec: {result.duration_sec:.1f}
    speaking_time_sec: {result.speaking_time_sec:.1f}
    words_per_minute: {result.words_per_minute:.1f}
    words_total: {result.words_total}
    fillers_count: {len(result.timeline.fillers)}
    fillers_per_100_words: {result.filler_words['per_100_words']:.1f}
    pauses_count: {len(result.timeline.pauses)}
    phrases_count: {len(result.timeline.phrases)}
    questions_count: {len(result.timeline.questions)}
    emphases_count: {len(result.timeline.emphases)}
    problems_count: {len(result.timeline.suspicious_moments)}

    Короткие списки примеров (если есть):
    Problems:
    {chr(10).join(problems_summary) if problems_summary else 'none'}
    Fillers (examples):
    {chr(10).join(filler_summary) if filler_summary else 'none'}
    Long pauses (examples):
    {chr(10).join(pause_summary) if pause_summary else 'none'}
    Emphases (examples):
    {chr(10).join(emphasis_summary) if emphasis_summary else 'none'}

    Требуется: вернуть JSON с полями:
     - overall_assessment: краткая оценка (строка)
     - strengths: массив кратких пунктов (строки)
     - areas_for_improvement: массив кратких пунктов (строки)
     - detailed_recommendations: массив объектов вида {{
         "timestamp":  float (секунды) | null,
         "duration": float | null,
         "type": one of ["pause","filler","phrasing","tempo","emphasis"],
         "action": краткое действие (например, "сократить паузу до 0.5с", "заменить 'типа' на 'например'"),
         "rationale": краткое объяснение, почему это поможет,
         "confidence": 0..1
       }}
     - key_insights: массив коротких наблюдений
     - confidence_score: число 0..1

    Правила и требования:
     - Будь прагматичен и избегай противоречивых советов. Если даёшь рекомендацию по паузам, уточняй какие именно (структурные vs. заполняющие).
     - Для слов-паразитов указывай конкретную замену: короткая пауза (0.2-0.6с), связующая фраза, или удаление.
     - Для темпа речи указывай, как менять паузы/скорость (например: "уменьшить среднюю паузу на 20%" или "увеличить паузу между ключевыми фразами до 0.8с").
     - Для каждого совета давай короткую причину и уровень уверенности (0..1).
     - Результат должен быть только JSON, без пояснительного текста.

    Транскрипт (сокращён):
    {result.transcript[:2000]}...

    """

        return prompt
