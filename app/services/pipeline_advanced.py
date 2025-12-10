"""
Продвинутый пайплайн с поддержкой детализированных таймингов.
"""
import logging
from typing import Optional
from pathlib import Path
import json

from fastapi import UploadFile

from app.services.transcriber import LocalWhisperTranscriber
from app.services.analyzer_advanced import AdvancedSpeechAnalyzer
from app.services.gigachat import GigaChatClient
from app.services.pipeline import SpeechAnalysisPipeline as BasePipeline
from app.models.timed_models import TimedAnalysisResult

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

            # 3. Продвинутый анализ с таймингами
            result = self.advanced_analyzer.analyze_with_timings(transcript)

            # 4. GigaChat анализ (если включен)
            if self.gigachat_client:
                result = await self._enhance_with_gigachat_advanced(result)

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
                f"- {question.timestamp:.1f}с: '{
                    question.text[:50]}...' ({q_type})"
            )

        # Акценты
        emphasis_summary = []
        for emphasis in result.timeline.emphases[:5]:
            emphasis_summary.append(
                f"- {emphasis.timestamp:.1f}с: {emphasis.type} - '{emphasis.description}'"
            )

        prompt = f"""
Проанализируй это публичное выступление с учетом детализированных таймингов:

=== ОСНОВНЫЕ МЕТРИКИ ===
Длительность: {result.duration_sec:.1f} секунд
Время говорения: {result.speaking_time_sec:.1f} секунд
Темп речи: {result.words_per_minute:.1f} слов/минуту
Слов всего: {result.words_total}
Слов-паразитов: {len(result.timeline.fillers)} ({result.filler_words['per_100_words']:.1f} на 100 слов)
Пауз: {len(result.timeline.pauses)}
Фраз: {len(result.timeline.phrases)}
Вопросов: {len(result.timeline.questions)}
Акцентов: {len(result.timeline.emphases)}
Проблемных моментов: {len(result.timeline.suspicious_moments)}

=== ПРОБЛЕМНЫЕ МОМЕНТЫ ===
{chr(10).join(problems_summary) if problems_summary else "Нет критических проблем"}

=== СЛОВА-ПАРАЗИТЫ ===
{chr(10).join(filler_summary) if filler_summary else "Нет слов-паразитов"}

=== ДЛИННЫЕ ПАУЗЫ ===
{chr(10).join(pause_summary) if pause_summary else "Нет длинных пауз"}

=== ВОПРОСЫ ===
{chr(10).join(question_summary) if question_summary else "Нет вопросов"}

=== АКЦЕНТЫ ===
{chr(10).join(emphasis_summary) if emphasis_summary else "Нет акцентов"}

=== ТРАНСКРИПТ ===
{result.transcript[:2000]}... [текст сокращен]

Дай развернутый анализ, уделяя внимание:
1. Временным паттернам (когда возникают проблемы)
2. Распределению слов-паразитов во времени
3. Ритму и темпу в разных частях выступления
4. Структуре фраз и пауз
5. Использованию вопросов и акцентов
6. Рекомендациям по конкретным временным отрезкам

Верни ответ в формате JSON с такими полями:
- overall_assessment: общая оценка выступления
- strengths: сильные стороны
- areas_for_improvement: зоны роста
- detailed_recommendations: конкретные рекомендации
- key_insights: ключевые инсайты
- confidence_score: уверенность анализа (0-1)
"""

        return prompt
