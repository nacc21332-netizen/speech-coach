"""
Продвинутый анализатор речи с детализированными таймингами.
"""
import logging
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

from app.models.transcript import Transcript, WordTiming
from app.models.timed_models import (
    TimedAnalysisResult, SpeechTimeline, WordTiming as AdvancedWordTiming,
    FillerWordDetail, PauseDetail, PhraseDetail, SuspiciousMoment,
    EmphasisDetail, QuestionDetail, SpeechElementType
)
from app.services.analyzer import (
    SpeechAnalyzer, FILLER_DEFINITIONS, COMPILED_FILLERS,
    MIN_PAUSE_GAP_SEC, LONG_PAUSE_SEC, SPEECH_RATE_WINDOW_SIZE,
    SPEECH_RATE_WINDOW_STEP
)
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PhraseBoundary:
    """Граница фразы"""
    index: int
    timestamp: float
    pause_duration: float
    confidence: float


class AdvancedSpeechAnalyzer:
    """Продвинутый анализатор речи с полными таймингами"""

    def __init__(self, analyzer: Optional[SpeechAnalyzer] = None):
        self.base_analyzer = analyzer or SpeechAnalyzer()

        # Улучшенный словарь слов-паразитов
        self.filler_patterns = [
            (name, pattern) for name, pattern in COMPILED_FILLERS
        ]

        # Пороги для классификации
        self.pause_thresholds = {
            "natural": 0.3,
            "dramatic": 1.0,
            "long": 2.5,
            "awkward": 4.0
        }

        # Пороги WPM
        self.wpm_thresholds = {
            "very_slow": 90,
            "slow": 120,
            "normal": 160,
            "fast": 200,
            "very_fast": 250
        }

    async def analyze_with_timings(self, transcript: Transcript, audio_path: Optional[Path] = None) -> TimedAnalysisResult:
        """
        Выполняет детализированный анализ речи с полными таймингами.
        """
        if not transcript.word_timings:
            logger.warning("No word timings in transcript")
            return self._create_empty_timed_result(transcript)

        # Базовый анализ
        enhanced_result = await self.base_analyzer.analyze(
            transcript, include_timings=True)

        # Собираем все слова с расширенной информацией (опционально используя аудио)
        advanced_words = self._create_advanced_word_timings(transcript, audio_path)

        # Анализируем элементы речи
        fillers = self._analyze_fillers(advanced_words, transcript)
        pauses = self._analyze_pauses(advanced_words)
        phrases = self._analyze_phrases(advanced_words, pauses)
        questions = self._analyze_questions(advanced_words)
        emphases = self._analyze_emphases(advanced_words)
        suspicious_moments = self._analyze_suspicious_moments(
            advanced_words, fillers, pauses, phrases
        )

        # Создаем временную шкалу
        timeline = SpeechTimeline(
            words=advanced_words,
            fillers=fillers,
            pauses=pauses,
            phrases=phrases,
            questions=questions,
            emphases=emphases,
            suspicious_moments=suspicious_moments
        )

        # Данные для визуализации
        speech_activity = self._build_speech_activity(advanced_words)
        speech_rate_windows = self._calculate_speech_rate_windows(
            advanced_words)
        intensity_profile = self._build_intensity_profile(
            advanced_words, emphases)
        emotion_timeline = self._build_emotion_timeline(
            advanced_words, emphases)

        # Общие метрики
        total_duration = max((w.end for w in advanced_words), default=0)
        speaking_time = sum(
            (w.duration for w in advanced_words)) if advanced_words else 0
        speaking_ratio = speaking_time / total_duration if total_duration > 0 else 0

        return TimedAnalysisResult(
            duration_sec=round(total_duration, 2),
            speaking_time_sec=round(speaking_time, 2),
            speaking_ratio=round(speaking_ratio, 3),
            words_total=len(advanced_words),
            words_per_minute=enhanced_result.words_per_minute,

            # Статистика
            filler_words={
                "total": len(fillers),
                "per_100_words": round(len(fillers) / len(advanced_words) * 100, 2) if advanced_words else 0.0,
                "items": self._group_fillers_by_type(fillers),
                "distribution": self._analyze_filler_distribution(fillers, total_duration)
            },
            pauses={
                "total": len(pauses),
                "avg_sec": sum(p.duration for p in pauses) / len(pauses) if pauses else 0,
                "max_sec": max((p.duration for p in pauses), default=0),
                "problematic_count": len([p for p in pauses if p.is_excessive]),
                "distribution": self._analyze_pause_distribution(pauses)
            },
            phrases={
                "total": len(phrases),
                "avg_words": sum(p.word_count for p in phrases) / len(phrases) if phrases else 0,
                "avg_duration_sec": sum(p.duration for p in phrases) / len(phrases) if phrases else 0,
                "complexity_score": self._calculate_phrase_complexity(phrases),
                "rhythm_score": self._calculate_rhythm_score(phrases)
            },

            # Советы из базового анализа
            advice=[item.dict() for item in enhanced_result.advice],

            # Транскрипт
            transcript=transcript.text,

            # Детализированные данные
            timeline=timeline,
            speech_activity=speech_activity,
            speech_rate_windows=speech_rate_windows,
            intensity_profile=intensity_profile,

            # GigaChat анализ (будет добавлен позже)
            gigachat_analysis=None,

            # Метаданные
            metadata={
                "analysis_version": "2.0",
                "includes_timings": True,
                "includes_visualization": True,
                "timestamp": None,
                "processing_time_sec": None
            }
        )

    def _create_advanced_word_timings(self, transcript: Transcript, audio_path: Optional[Path] = None) -> List[AdvancedWordTiming]:
        """Создает расширенные тайминги слов"""
        advanced_words = []

        # Precompute audio info if available
        audio_info = None
        if audio_path is not None:
            try:
                import wave
                import numpy as _np

                wf = wave.open(str(audio_path), 'rb')
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_info = dict(wf=wf, n_channels=n_channels, sampwidth=sampwidth, framerate=framerate, n_frames=n_frames)
            except Exception as e:
                logger.warning(f"Could not read audio for RMS computation: {e}")
                audio_info = None

        for i, word in enumerate(transcript.word_timings):
            # Контекст
            context_before = self._get_context(
                transcript.word_timings, i, -2, 0)
            context_after = self._get_context(transcript.word_timings, i, 0, 2)

            # Проверка на слово-паразит - используем search вместо match
            word_text = word.word.lower().strip().strip(",.!?;:()\"'")
            is_filler = any(
                pattern.search(word_text)
                for _, pattern in self.filler_patterns
            )

            # Проверка на колебание
            is_hesitation = self._is_hesitation_word(word.word)



            # compute rms if audio available
            word_rms = None
            if audio_info is not None:
                try:
                    wf = audio_info['wf']
                    fr = audio_info['framerate']
                    start_frame = int(max(0, math.floor(word.start * fr)))
                    end_frame = int(max(start_frame + 1, math.ceil(word.end * fr)))
                    wf.setpos(start_frame)
                    raw = wf.readframes(max(0, end_frame - start_frame))
                    import numpy as _np
                    if audio_info['sampwidth'] == 2:
                        dtype = _np.int16
                    elif audio_info['sampwidth'] == 4:
                        dtype = _np.int32
                    else:
                        dtype = _np.int16
                    arr = _np.frombuffer(raw, dtype=dtype)
                    # if stereo, take mean across channels
                    if audio_info['n_channels'] > 1:
                        arr = arr.reshape(-1, audio_info['n_channels']).mean(axis=1)
                    if arr.size > 0:
                        word_rms = float(_np.sqrt(_np.mean(arr.astype(_np.float64) ** 2)))
                    else:
                        word_rms = 0.0
                except Exception:
                    word_rms = None

            advanced_words.append(AdvancedWordTiming(
                word=word.word,
                start=word.start,
                end=word.end,
                duration=word.end - word.start if word.end and word.start else 0,
                confidence=word.confidence,
                type=SpeechElementType.WORD,
                is_filler=is_filler,
                is_hesitation=is_hesitation,
                context_before=context_before,
                context_after=context_after
                ,
                rms=word_rms
            ))

        # close wave if opened
        try:
            if audio_info is not None and 'wf' in audio_info and audio_info['wf']:
                audio_info['wf'].close()
        except Exception:
            pass

        return advanced_words

    def _analyze_fillers(self, words: List[AdvancedWordTiming],
                         transcript: Transcript) -> List[FillerWordDetail]:
        """Анализирует слова-паразиты с детализацией"""
        fillers = []
        filler_counter = Counter()

        for i, word in enumerate(words):
            if word.is_filler:
                # Определяем тип слова-паразита
                filler_type = self._identify_filler_type(word.word)

                # Контекстное окно
                context_window = self._get_context_window(words, i, -2, 2)

                # Частота
                filler_counter[filler_type] += 1
                frequency = filler_counter[filler_type]

                # Серьезность
                severity = self._calculate_filler_severity(
                    filler_type, frequency)

                # Рекомендации
                suggestions = self._generate_filler_suggestions(
                    filler_type, context_window)

                fillers.append(FillerWordDetail(
                    word=filler_type,
                    timestamp=word.start,
                    exact_word=word.word,
                    duration=word.duration,
                    confidence=word.confidence,
                    frequency=frequency,
                    context_window=context_window,
                    segment_index=i,
                    severity=severity,
                    suggestions=suggestions
                ))

        return fillers

    def _analyze_pauses(self, words: List[AdvancedWordTiming]) -> List[PauseDetail]:
        """Анализирует паузы между словами"""
        pauses = []

        if len(words) < 2:
            return pauses

        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]

            pause_start = current_word.end
            pause_end = next_word.start
            pause_duration = pause_end - pause_start

            if pause_duration >= MIN_PAUSE_GAP_SEC:
                # Контекст
                context_before = current_word.context_before or ""
                context_after = next_word.context_after or ""

                # Определяем тип паузы
                pause_type = self._classify_pause_type(pause_duration)

                # Проверяем на избыточность
                is_excessive = pause_duration > LONG_PAUSE_SEC

                # Рекомендация
                recommendation = self._generate_pause_recommendation(
                    pause_duration, pause_type, context_before, context_after
                )

                pauses.append(PauseDetail(
                    start=pause_start,
                    end=pause_end,
                    duration=pause_duration,
                    type=pause_type,
                    intensity=min(pause_duration / 3.0, 1.0),
                    context_before=context_before,
                    context_after=context_after,
                    before_word=current_word.word,
                    after_word=next_word.word,
                    is_excessive=is_excessive,
                    natural_threshold=LONG_PAUSE_SEC,
                    recommendation=recommendation if is_excessive else None
                ))

        return pauses

    def _analyze_phrases(self, words: List[AdvancedWordTiming],
                         pauses: List[PauseDetail]) -> List[PhraseDetail]:
        """Определяет и анализирует фразы"""
        phrases = []

        if not words:
            return phrases

        # Находим границы фраз на основе пауз
        phrase_boundaries = self._find_phrase_boundaries(words, pauses)

        # Создаем фразы
        phrase_start_idx = 0
        for phrase_id, boundary in enumerate(phrase_boundaries):
            phrase_words = words[phrase_start_idx:boundary.index + 1]

            if phrase_words:
                phrase = self._create_phrase(phrase_id, phrase_words, pauses)
                phrases.append(phrase)

            phrase_start_idx = boundary.index + 1

        # Последняя фраза
        if phrase_start_idx < len(words):
            phrase_words = words[phrase_start_idx:]
            phrase = self._create_phrase(len(phrases), phrase_words, pauses)
            phrases.append(phrase)

        return phrases

    def _analyze_questions(self, words: List[AdvancedWordTiming]) -> List[QuestionDetail]:
        """Анализирует вопросы в речи"""
        questions = []
        question_words = {"кто", "что", "где", "когда",
                          "почему", "как", "зачем", "ли", "разве", "неужели"}

        # Группируем слова в предложения
        sentences = self._group_into_sentences(words)

        for sentence in sentences:
            if any(word.word.lower() in question_words for word in sentence.words):
                # Это вопрос
                is_rhetorical = self._is_rhetorical_question(sentence.text)

                questions.append(QuestionDetail(
                    text=sentence.text,
                    timestamp=sentence.start,
                    duration=sentence.duration,
                    word_count=len(sentence.words),
                    is_rhetorical=is_rhetorical,
                    confidence=0.8 if is_rhetorical else 0.9,
                    answer_prompted=not is_rhetorical,
                    effectiveness_score=self._calculate_question_effectiveness(
                        sentence)
                ))

        return questions

    def _analyze_emphases(self, words: List[AdvancedWordTiming]) -> List[EmphasisDetail]:
        """Анализирует акценты/эмоциональные моменты"""
        emphases = []

        # Use robust local-window statistics (median + MAD) to avoid global skew
        # and to detect emphases relative to local context rather than entire speech.
        def median(data: List[float]) -> float:
            s = sorted(data)
            n = len(s)
            if n == 0:
                return 0.0
            mid = n // 2
            return (s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0)

        def mad(data: List[float], med: float) -> float:
            return median([abs(x - med) for x in data])

        # Parameters from settings
        window_size = max(3, settings.emphasis_window_size)
        half_win = window_size // 2
        min_duration_for_emphasis = max(0.01, float(settings.emphasis_min_duration))
        mad_multiplier = float(settings.emphasis_mad_multiplier)
        pause_threshold = float(settings.emphasis_pause_threshold)
        content_boost = float(settings.emphasis_content_boost)

        for i, word in enumerate(words):
            # Build local window
            start_idx = max(0, i - half_win)
            end_idx = min(len(words), i + half_win + 1)
            local_durations = [w.duration for w in words[start_idx:end_idx] if w.duration > 0]

            if not local_durations:
                continue

            loc_med = median(local_durations)
            loc_mad = mad(local_durations, loc_med) or 1e-6

            # compute local RMS median/MAD if rms values exist
            local_rms_values = [w.rms for w in words[start_idx:end_idx] if getattr(w, 'rms', None) is not None]
            loc_rms_med = median(local_rms_values) if local_rms_values else None
            loc_rms_mad = mad(local_rms_values, loc_rms_med) if local_rms_values else None

            # Duration-based emphasis: require both absolute minimum and relative deviation
            if word.duration >= min_duration_for_emphasis and (word.duration - loc_med) > (mad_multiplier * loc_mad):
                # intensity scaled by how many MADs above median (clamped)
                z = (word.duration - loc_med) / loc_mad
                intensity = max(0.0, min((z - mad_multiplier) / (mad_multiplier * 2.0), 1.0))  # map z roughly to 0..1
                emphases.append(EmphasisDetail(
                    timestamp=word.start,
                    type="duration",
                    intensity=round(float(intensity), 3),
                    description=f"Длительное произнесение слова '{word.word}' (локально медиа {loc_med:.3f}s)",
                    context=self._get_context_window(words, i, -1, 1),
                    effectiveness=0.7,
                    suggestion="Используйте такие акценты для ключевых слов"
                ))

            # Volume-based emphasis using per-word RMS if available
            if getattr(word, 'rms', None) is not None and loc_rms_med is not None and loc_rms_mad:
                try:
                    if (word.rms - loc_rms_med) > (mad_multiplier * loc_rms_mad):
                        z_r = (word.rms - loc_rms_med) / (loc_rms_mad or 1e-6)
                        vol_intensity = max(0.0, min((z_r - mad_multiplier) / (mad_multiplier * 2.0), 1.0))
                        emphases.append(EmphasisDetail(
                            timestamp=word.start,
                            type="volume",
                            intensity=round(float(vol_intensity), 3),
                            description=f"Повышенная громкость слова '{word.word}'",
                            context=self._get_context_window(words, i, -1, 1),
                            effectiveness=0.8,
                            suggestion="Используйте контролируемую громкость для акцента"
                        ))
                except Exception:
                    pass

            # Repetition (adjacent duplicate) — keep but avoid duplicates
            if i > 0 and word.word.lower() == words[i-1].word.lower():
                # Only add if not already captured by duration or other repetition
                emphases.append(EmphasisDetail(
                    timestamp=word.start,
                    type="repetition",
                    intensity=0.8,
                    description=f"Повтор слова '{word.word}'",
                    context=self._get_context_window(words, i, -2, 2),
                    effectiveness=0.6,
                    suggestion="Повтор может быть эффективным, но не злоупотребляйте им"
                ))

            # Content-based emphasis words (lexical cues) — only if they stand out locally
            emphasis_words = {
                "очень": 0.6, "важно": 0.8, "критично": 0.8, "серьезно": 0.7,
                "особенно": 0.7, "прежде": 0.7, "всего": 0.7, "именно": 0.7, "как": 0.6, "раз": 0.6, "так": 0.6, "вот": 0.6,
                "разумеется": 0.7, "безусловно": 0.8, "обязательно": 0.8, "абсолютно": 0.7, "наверное": 0.5, "кажется": 0.5
            }
            if word.word.lower() in emphasis_words:
                # content emphasis: stronger if duration is above local median or pause before
                pause_before = 0.0
                if i > 0:
                    prev = words[i-1]
                    pause_before = max(0.0, word.start - prev.end)

                score = emphasis_words[word.word.lower()]
                # boost if longer than local median or preceded by a pause
                boost = 0.0
                if word.duration > loc_med:
                    boost += min((word.duration - loc_med) / max(loc_med, 0.01), 1.0)
                if pause_before > 0.5:
                    boost += min(pause_before / 2.0, 0.5)

                intensity = min(score + content_boost * boost, 1.0)
                emphases.append(EmphasisDetail(
                    timestamp=word.start,
                    type="content",
                    intensity=round(float(intensity), 3),
                    description=f"Усиленное слово '{word.word}'",
                    context=self._get_context_window(words, i, -1, 1),
                    effectiveness=0.8,
                    suggestion="Хорошее использование эмоционального акцента"
                ))

            # Pause-before emphasis
            if i > 0:
                prev_word = words[i-1]
                pause_before = word.start - prev_word.end
                if pause_before > pause_threshold:
                    emphases.append(EmphasisDetail(
                        timestamp=word.start,
                        type="pause",
                        intensity=min(pause_before / 2.0, 1.0),
                        description=f"Акцент через паузу перед словом '{word.word}'",
                        context=self._get_context_window(words, i, -1, 1),
                        effectiveness=0.9,
                        suggestion="Хорошее использование паузы для акцента"
                    ))

            # Speed-based emphasis: check deviation from local median duration
            if i > 0 and i < len(words) - 1:
                relative = word.duration / loc_med if loc_med > 0 else 1.0
                if relative < 0.6 or relative > 1.6:
                    speed_type = "speed" if relative < 1.0 else "duration"
                    speed_intensity = min(abs(relative - 1.0), 1.0)
                    emphases.append(EmphasisDetail(
                        timestamp=word.start,
                        type=speed_type,
                        intensity=round(float(speed_intensity), 3),
                        description=f"Изменение темпа речи на слове '{word.word}'",
                        context=self._get_context_window(words, i, -1, 1),
                        effectiveness=0.7,
                        suggestion="Изменение темпа может использоваться для акцента"
                    ))

        return emphases

    def _analyze_suspicious_moments(self, words: List[AdvancedWordTiming],
                                    fillers: List[FillerWordDetail],
                                    pauses: List[PauseDetail],
                                    phrases: List[PhraseDetail]) -> List[SuspiciousMoment]:
        """Находит сомнительные/проблемные моменты"""
        moments = []
        moment_id = 0

        # 1. Скопление слов-паразитов
        filler_clusters = self._find_filler_clusters(fillers)
        for cluster in filler_clusters:
            moments.append(
                self._create_filler_cluster_moment(moment_id, cluster))
            moment_id += 1

        # 2. Длинные паузы
        for pause in pauses:
            if pause.is_excessive:
                moments.append(
                    self._create_long_pause_moment(moment_id, pause))
                moment_id += 1

        # 3. Быстрая/медленная речь
        for phrase in phrases:
            if phrase.words_per_minute > 200:  # Очень быстро
                moments.append(
                    self._create_fast_speech_moment(moment_id, phrase))
                moment_id += 1
            elif phrase.words_per_minute < 80:  # Очень медленно
                moments.append(
                    self._create_slow_speech_moment(moment_id, phrase))
                moment_id += 1

        # 4. Сложные фразы
        for phrase in phrases:
            if phrase.is_complex:
                moments.append(
                    self._create_complex_phrase_moment(moment_id, phrase))
                moment_id += 1

        # 5. Моменты колебания/нерешительности
        hesitation_moments = self._find_hesitation_moments(words, pauses)
        for hesitation in hesitation_moments:
            moments.append(hesitation)
            moment_id += 1

        return moments

    def _find_hesitation_moments(self, words: List[AdvancedWordTiming], 
                                 pauses: List[PauseDetail]) -> List[SuspiciousMoment]:
        """Находит моменты колебания/нерешительности"""
        moments = []
        moment_id = 0
        
        # 1. Проверяем слова с признаками колебания
        for i, word in enumerate(words):
            if word.is_hesitation:
                moments.append(SuspiciousMoment(
                    id=moment_id,
                    timestamp=word.start,
                    type="hesitation",
                    severity="medium",
                    duration=word.duration,
                    description=f"Колебание/нерешительность: '{word.word}'",
                    suggestion="Избегайте звуков нерешительности, делайте короткую паузу вместо 'эээ', 'ааа'",
                    context_before=word.context_before or "",
                    context_after=word.context_after or "",
                    confidence=0.8,
                    words_affected=[word.word],
                    improvement_potential=0.7
                ))
                moment_id += 1
        
        # 2. Проверяем частые короткие паузы подряд (признак нерешительности)
        for i in range(len(pauses) - 1):
            current_pause = pauses[i]
            next_pause = pauses[i + 1]
            
            # Если между паузами мало слов - это может быть нерешительность
            words_between = [w for w in words 
                           if current_pause.end <= w.start and w.end <= next_pause.start]
            
            if len(words_between) <= 1 and abs(current_pause.end - next_pause.start) < 2.0:
                # Две короткие паузы близко друг к другу
                if current_pause.duration < 1.0 and next_pause.duration < 1.0:
                    moments.append(SuspiciousMoment(
                        id=moment_id,
                        timestamp=current_pause.start,
                        type="hesitation",
                        severity="medium",
                        duration=current_pause.duration + next_pause.duration,
                        description="Частые короткие паузы, признак нерешительности",
                        suggestion="Стремитесь к более плавному потоку речи, избегайте частых остановок",
                        context_before=current_pause.context_before,
                        context_after=next_pause.context_after,
                        confidence=0.6,
                        words_affected=[current_pause.before_word, next_pause.before_word] 
                                    if current_pause.before_word and next_pause.before_word else [],
                        improvement_potential=0.6
                    ))
                    moment_id += 1
        
        # 3. Проверяем слова-паразиты, идущие подряд
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            if current_word.is_filler and next_word.is_filler:
                time_gap = next_word.start - current_word.end
                if time_gap < 1.0:  # Слова-паразиты идут подряд
                    moments.append(SuspiciousMoment(
                        id=moment_id,
                        timestamp=current_word.start,
                        type="hesitation",
                        severity="high",
                        duration=time_gap + current_word.duration + next_word.duration,
                        description=f"Скопление слов-паразитов: '{current_word.word}' и '{next_word.word}'",
                        suggestion="Замените скопления слов-паразитов на короткую паузу или переходное выражение",
                        context_before=current_word.context_before or "",
                        context_after=next_word.context_after or "",
                        confidence=0.8,
                        words_affected=[current_word.word, next_word.word],
                        improvement_potential=0.8
                    ))
                    moment_id += 1
        
        return moments

    def _create_phrase(self, phrase_id: int, words: List[AdvancedWordTiming],
                       pauses: List[PauseDetail]) -> PhraseDetail:
        """Создает детализированную фразу"""
        if not words:
            return PhraseDetail(
                id=phrase_id,
                start=0,
                end=0,
                duration=0,
                word_count=0,
                words_per_minute=0,
                text="",
                words=[],
                filler_count=0,
                pause_count=0,
                avg_pause_duration=0,
                complexity_score=0,
                is_complex=False,
                clarity_score=0
            )

        start = words[0].start
        end = words[-1].end
        duration = end - start
        text = " ".join(w.word for w in words)
        word_count = len(words)

        # Слова-паразиты в фразе
        filler_count = sum(1 for w in words if w.is_filler)

        # Паузы внутри фразы
        phrase_pauses = [
            p for p in pauses
            if p.start >= start and p.end <= end
        ]
        pause_count = len(phrase_pauses)
        avg_pause_duration = (
            sum(p.duration for p in phrase_pauses) / pause_count
            if pause_count > 0 else 0
        )

        # Темп речи
        speaking_time = duration - sum(p.duration for p in phrase_pauses)
        words_per_minute = (
            word_count / (speaking_time / 60)
            if speaking_time > 0 else 0
        )

        # Сложность
        complexity_score = self._calculate_phrase_complexity_score(words)
        is_complex = complexity_score > 0.7

        # Ясность
        clarity_score = self._calculate_clarity_score(
            words, avg_pause_duration)



        return PhraseDetail(
            id=phrase_id,
            start=start,
            end=end,
            duration=duration,
            word_count=word_count,
            words_per_minute=words_per_minute,
            text=text,
            words=words,
            filler_count=filler_count,
            pause_count=pause_count,
            avg_pause_duration=avg_pause_duration,
            complexity_score=complexity_score,
            is_complex=is_complex,
            clarity_score=clarity_score
        )

    # Вспомогательные методы
    def _get_context(self, words: List[WordTiming], index: int,
                     before: int, after: int) -> str:
        """Получает контекст вокруг слова"""
        start = max(0, index + before)
        end = min(len(words), index + after + 1)
        context_words = words[start:end]
        return " ".join(w.word for w in context_words)

    def _get_context_window(self, words: List[AdvancedWordTiming],
                            index: int, before: int, after: int) -> str:
        """Получает контекстное окно"""
        start = max(0, index + before)
        end = min(len(words), index + after + 1)
        context_words = words[start:end]

        # Помечаем центральное слово
        result = []
        for i, word in enumerate(context_words):
            if start + i == index:
                result.append(f"[{word.word}]")
            else:
                result.append(word.word)

        return " ".join(result)

    def _is_hesitation_word(self, word: str) -> bool:
        """Проверяет, является ли слово колебанием"""
        hesitation_patterns = [
            r'э+', r'а+', r'мм+', r'гм+'
        ]
        word_lower = word.lower()
        return any(re.match(pattern, word_lower) for pattern in hesitation_patterns)

    def _identify_filler_type(self, word: str) -> str:
        """Идентифицирует тип слова-паразита"""
        word_lower = word.lower().strip().strip(",.!?;:()\"'")

        for filler_name, pattern in self.filler_patterns:
            # Используем search вместо match для лучшего совпадения
            if pattern.search(word_lower):
                return filler_name

        # Дополнительные проверки
        if re.search(r'э+', word_lower):
            return "э-э"
        elif re.search(r'а+', word_lower):
            return "а-а"
        elif word_lower in ["ну", "вот"]:
            return word_lower

        return "unknown"

    def _calculate_filler_severity(self, filler_type: str, frequency: int) -> str:
        """Определяет серьезность слова-паразита"""
        if frequency > 10:
            return "high"
        elif frequency > 5:
            return "medium"
        else:
            return "low"

    def _generate_filler_suggestions(self, filler_type: str, context: str) -> List[str]:
        """Генерирует рекомендации по словам-паразитам"""
        suggestions = []
        base_suggestions = {
            "э-э": ["Замените на короткую паузу (0.2–0.6с) или тишину, если это не влияет на смысл", "Сделайте вдох перед фразой"],
            "ну": ["Попробуйте заменить на связующую фразу ('итак', 'таким образом') или убрать, если слово не несет смысловой нагрузки"],
            "вот": ["Если слово не несет семантики — опустите его; если выступление требует указаний, замените на 'например' или укажи конкретно"],
            "как бы": ["Уберите 'как бы' в подготовленной речи; в репликах используйте короткую паузу"],
            "типа": ["Замените на 'например' или формулировку с уточнением"],
            "то есть": ["Используйте 'то есть' осознанно для пояснения; в остальных случаях опустите"]
        }

        if filler_type in base_suggestions:
            suggestions.extend(base_suggestions[filler_type])

        # General practical suggestion — conservative: replace by short pause OR transitional phrase.
        suggestions.append("Практикуйте замены: короткая пауза (0.2–0.6с) или связующая фраза; избегайте добавления дополнительных пауз, если паузы уже длинные в выступлении.")

        return suggestions

    def _classify_pause_type(self, duration: float) -> str:
        """Классифицирует паузу по длительности"""
        if duration < self.pause_thresholds["dramatic"]:
            return "natural"
        elif duration < self.pause_thresholds["long"]:
            return "dramatic"
        elif duration < self.pause_thresholds["awkward"]:
            return "long"
        else:
            return "awkward"

    def _generate_pause_recommendation(self, duration: float, pause_type: str,
                                       context_before: str, context_after: str) -> str:
        """Генерирует рекомендацию по паузе"""
        if pause_type == "awkward":
            return f"Пауза {duration:.1f}с слишком длинная. Сократите до 2-3 секунд."
        elif pause_type == "long":
            return f"Длинная пауза {duration:.1f}с. Рассмотрите добавление связующей фразы."
        else:
            return "Пауза в пределах нормы."

    def _find_phrase_boundaries(self, words: List[AdvancedWordTiming],
                                pauses: List[PauseDetail]) -> List[PhraseBoundary]:
        """Находит границы фраз"""
        boundaries = []

        # Используем паузы как границы
        for pause in pauses:
            # Находим индекс слова перед паузой
            for i, word in enumerate(words):
                if abs(word.end - pause.start) < 0.1:
                    confidence = min(pause.duration / 2.0, 1.0)
                    boundaries.append(PhraseBoundary(
                        index=i,
                        timestamp=word.end,
                        pause_duration=pause.duration,
                        confidence=confidence
                    ))
                    break

        return sorted(boundaries, key=lambda x: x.index)

    def _group_into_sentences(self, words: List[AdvancedWordTiming]) -> List[Any]:
        """Группирует слова в предложения"""
        sentences = []
        current_sentence = []

        for word in words:
            current_sentence.append(word)

            # Конец предложения
            if word.word.endswith(('.', '!', '?')):
                if current_sentence:
                    sentences.append(
                        self._create_sentence_object(current_sentence))
                    current_sentence = []

        # Последнее предложение
        if current_sentence:
            sentences.append(self._create_sentence_object(current_sentence))

        return sentences

    def _create_sentence_object(self, words: List[AdvancedWordTiming]) -> Any:
        """Создает объект предложения"""
        from dataclasses import dataclass

        @dataclass
        class Sentence:
            text: str
            start: float
            end: float
            duration: float
            words: List[AdvancedWordTiming]

        if not words:
            return Sentence("", 0, 0, 0, [])

        start = words[0].start
        end = words[-1].end
        text = " ".join(w.word for w in words)

        return Sentence(text, start, end, end - start, words)

    def _is_rhetorical_question(self, text: str) -> bool:
        """Определяет, является ли вопрос риторическим"""
        rhetorical_indicators = [
            "разве", "неужели", "ведь", "же",
            "можно ли", "стоит ли"
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in rhetorical_indicators)

    def _calculate_question_effectiveness(self, sentence: Any) -> float:
        """Рассчитывает эффективность вопроса"""
        score = 0.5

        # Короткие вопросы обычно эффективнее
        if len(sentence.words) <= 5:
            score += 0.2

        # Вопросы с "почему", "как" обычно эффективнее
        if any(w in sentence.text.lower() for w in ["почему", "как", "зачем"]):
            score += 0.1

        return min(score, 1.0)

    def _find_filler_clusters(self, fillers: List[FillerWordDetail]) -> List[List[FillerWordDetail]]:
        """Находит скопления слов-паразитов"""
        if not fillers:
            return []

        clusters = []
        current_cluster = []

        for i, filler in enumerate(fillers):
            if not current_cluster:
                current_cluster.append(filler)
            else:
                last_filler = current_cluster[-1]
                time_gap = filler.timestamp - last_filler.timestamp

                if time_gap < settings.filler_cluster_gap_sec:  # Слова-паразиты в пределах настроенного окна кластеризации
                    current_cluster.append(filler)
                else:
                    if len(current_cluster) >= 2:  # Кластер минимум из 2 слов
                        clusters.append(current_cluster)
                    current_cluster = [filler]

        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        return clusters

    def _create_filler_cluster_moment(self, moment_id: int,
                                      cluster: List[FillerWordDetail]) -> SuspiciousMoment:
        """Создает момент для скопления слов-паразитов"""
        words = [f.word for f in cluster]
        context_before = cluster[0].context_window if cluster else ""
        context_after = cluster[-1].context_window if cluster else ""

        if len(cluster) >= 5:
            severity = "critical"
        elif len(cluster) > 3:
            severity = "high"
        else:
            severity = "medium"

        return SuspiciousMoment(
            id=moment_id,
            timestamp=cluster[0].timestamp if cluster else 0,
            type="excessive_filler",
            severity=severity,
            duration=cluster[-1].timestamp -
            cluster[0].timestamp if len(cluster) > 1 else 0,
            description=f"Скопление {len(cluster)} слов-паразитов: {', '.join(words)}",
            suggestion="Сделайте паузу и глубокий вдох. Говорите медленнее и осознаннее.",
            context_before=context_before,
            context_after=context_after,
            confidence=0.9,
            words_affected=words,
            improvement_potential=0.8
        )

    def _create_long_pause_moment(self, moment_id: int,
                                  pause: PauseDetail) -> SuspiciousMoment:
        """Создает момент для длинной паузы"""
        return SuspiciousMoment(
            id=moment_id,
            timestamp=pause.start,
            type="long_pause",
            severity="medium",
            duration=pause.duration,
            description=f"Длинная пауза {pause.duration:.1f} секунд",
            suggestion=f"Сократите паузу до 2-3 секунд. Добавьте связующую фразу.",
            context_before=pause.context_before,
            context_after=pause.context_after,
            confidence=0.8,
            words_affected=[
                pause.before_word, pause.after_word] if pause.before_word and pause.after_word else [],
            improvement_potential=0.7
        )

    def _create_fast_speech_moment(self, moment_id: int,
                                   phrase: PhraseDetail) -> SuspiciousMoment:
        """Создает момент для быстрой речи"""
        return SuspiciousMoment(
            id=moment_id,
            timestamp=phrase.start,
            type="fast_speech",
            severity="medium",
            duration=phrase.duration,
            description=f"Очень быстрый темп: {phrase.words_per_minute:.0f} слов в минуту",
            suggestion="Замедлите речь. Делайте паузы между фразами.",
            context_before=phrase.text[:50],
            context_after="",
            confidence=0.7,
            words_affected=[w.word for w in phrase.words[:3]],
            improvement_potential=0.6
        )

    def _create_slow_speech_moment(self, moment_id: int,
                                   phrase: PhraseDetail) -> SuspiciousMoment:
        """Создает момент для медленной речи"""
        return SuspiciousMoment(
            id=moment_id,
            timestamp=phrase.start,
            type="slow_speech",
            severity="low",
            duration=phrase.duration,
            description=f"Очень медленный темп: {phrase.words_per_minute:.0f} слов в минуту",
            suggestion="Ускорьте немного темп. Аудитория может потерять интерес.",
            context_before=phrase.text[:50],
            context_after="",
            confidence=0.7,
            words_affected=[w.word for w in phrase.words[:3]],
            improvement_potential=0.5
        )

    def _create_complex_phrase_moment(self, moment_id: int,
                                      phrase: PhraseDetail) -> SuspiciousMoment:
        """Создает момент для сложной фразы"""
        return SuspiciousMoment(
            id=moment_id,
            timestamp=phrase.start,
            type="incoherence",
            severity="medium",
            duration=phrase.duration,
            description=f"Слишком сложная фраза: {phrase.word_count} слов",
            suggestion="Разбейте на несколько более простых фраз.",
            context_before=phrase.text[:50],
            context_after="",
            confidence=0.6,
            words_affected=[w.word for w in phrase.words[:5]],
            improvement_potential=0.7
        )

    def _build_speech_activity(self, words: List[AdvancedWordTiming],
                               resolution: float = 0.1) -> List[Dict[str, float]]:
        """Строит данные активности речи для визуализации"""
        if not words:
            return []

        total_duration = max(w.end for w in words)
        activity = []
        current_time = 0.0

        while current_time <= total_duration:
            is_speaking = 0.0

            for word in words:
                if word.start <= current_time <= word.end:
                    is_speaking = 1.0
                    break

            activity.append({
                "time": round(current_time, 2),
                "is_speaking": is_speaking
            })

            current_time += resolution

        return activity

    def _calculate_speech_rate_windows(self, words: List[AdvancedWordTiming]) -> List[Dict[str, Any]]:
        """Рассчитывает темп речи в окнах"""
        if not words:
            return []

        total_duration = max(w.end for w in words)
        windows = []
        current_start = 0.0

        while current_start < total_duration:
            window_end = min(
                current_start + SPEECH_RATE_WINDOW_SIZE, total_duration)

            # Считаем слова и время говорения в окне
            word_count = 0
            speaking_time = 0.0

            for word in words:
                # Слово полностью в окне
                if word.start >= current_start and word.end <= window_end:
                    word_count += 1
                    speaking_time += word.duration
                # Слово частично в окне
                elif word.start < window_end and word.end > current_start:
                    overlap_start = max(word.start, current_start)
                    overlap_end = min(word.end, window_end)
                    if overlap_end > overlap_start:
                        # Пропорционально учитываем слово
                        word_fraction = (
                            overlap_end - overlap_start) / word.duration
                        word_count += word_fraction
                        speaking_time += (overlap_end - overlap_start)

            wpm = word_count / \
                (speaking_time / 60.0) if speaking_time > 0 else 0.0

            windows.append({
                "window_start": round(current_start, 2),
                "window_end": round(window_end, 2),
                "word_count": round(word_count, 1),
                "words_per_minute": round(wpm, 1),
                "speaking_time": round(speaking_time, 2)
            })

            current_start += SPEECH_RATE_WINDOW_STEP

        return windows

    def _build_intensity_profile(self, words: List[AdvancedWordTiming],
                                 emphases: List[EmphasisDetail]) -> List[Dict[str, float]]:
        """Строит профиль интенсивности речи"""
        if not words:
            return []

        total_duration = max(w.end for w in words)
        profile = []
        resolution = 0.5  # секунда

        current_time = 0.0
        while current_time <= total_duration:
            intensity = 0.0

            # Базовая интенсивность от слов
            for word in words:
                if word.start <= current_time <= word.end:
                    # Длительные слова имеют большую интенсивность
                    intensity = min(word.duration * 2, 1.0)
                    break

            # Усиление от акцентов
            for emphasis in emphases:
                if abs(emphasis.timestamp - current_time) < 1.0:
                    intensity = max(intensity, emphasis.intensity)

            profile.append({
                "time": round(current_time, 2),
                "intensity": round(intensity, 2)
            })

            current_time += resolution

        return profile

    def _build_emotion_timeline(self, words: List[AdvancedWordTiming],
                                emphases: List[EmphasisDetail]) -> List[Dict[str, Any]]:
        """Строит временную шкалу эмоциональности (теперь пустую, т.к. эмоциональность не определяется по аудио)"""
        return []

    def _group_fillers_by_type(self, fillers: List[FillerWordDetail]) -> List[Dict[str, Any]]:
        """Группирует слова-паразиты по типу"""
        from collections import Counter
        counter = Counter(f.word for f in fillers)

        return [
            {
                "word": word,
                "count": count,
                "percentage": round(count / len(fillers) * 100, 2) if fillers else 0.0
            }
            for word, count in counter.most_common()
        ]

    def _analyze_filler_distribution(self, fillers: List[FillerWordDetail],
                                     total_duration: float) -> Dict[str, Any]:
        """Анализирует распределение слов-паразитов по времени"""
        if not fillers or total_duration <= 0:
            return {}

        # Разделяем на трети
        third = total_duration / 3
        first_third = len([f for f in fillers if f.timestamp < third])
        second_third = len(
            [f for f in fillers if third <= f.timestamp < 2*third])
        third_third = len([f for f in fillers if f.timestamp >= 2*third])

        return {
            "first_third": first_third,
            "second_third": second_third,
            "third_third": third_third,
            "most_common_zone": "начало" if first_third > second_third and first_third > third_third else
            "середина" if second_third > third_third else "конец"
        }

    def _analyze_pause_distribution(self, pauses: List[PauseDetail]) -> Dict[str, Any]:
        """Анализирует распределение пауз"""
        if not pauses:
            return {}

        # Группируем по типу
        type_counts = {}
        for pause in pauses:
            type_counts[pause.type] = type_counts.get(pause.type, 0) + 1

        # Средняя длительность по типам
        type_durations = {}
        for pause in pauses:
            if pause.type not in type_durations:
                type_durations[pause.type] = []
            type_durations[pause.type].append(pause.duration)

        avg_durations = {
            ptype: sum(durs) / len(durs)
            for ptype, durs in type_durations.items()
        }

        return {
            "by_type": type_counts,
            "avg_duration_by_type": avg_durations,
            "excessive_ratio": len([p for p in pauses if p.is_excessive]) / len(pauses)
        }

    def _calculate_phrase_complexity(self, phrases: List[PhraseDetail]) -> float:
        """Рассчитывает общую сложность фраз"""
        if not phrases:
            return 0.0

        return sum(p.complexity_score for p in phrases) / len(phrases)

    def _calculate_rhythm_score(self, phrases: List[PhraseDetail]) -> float:
        """Рассчитывает оценку ритма"""
        if len(phrases) < 2:
            return 0.5

        # Вариативность длительности фраз
        durations = [p.duration for p in phrases]
        avg_duration = sum(durations) / len(durations)

        if avg_duration == 0:
            return 0.5

        # Коэффициент вариации
        variance = sum((d - avg_duration) **
                       2 for d in durations) / len(durations)
        std = math.sqrt(variance)
        cv = std / avg_duration

        # Преобразуем в оценку 0-1
        # cv < 0.3 - слишком монотонно, cv > 0.7 - слишком хаотично
        # Идеально около 0.5
        return max(0.0, min(1.0, 1 - abs(cv - 0.5)))

    def _calculate_phrase_complexity_score(self, words: List[AdvancedWordTiming]) -> float:
        """Рассчитывает сложность фразы"""
        if not words:
            return 0.0

        score = 0.0

        # Количество слов
        word_count = len(words)
        if word_count > 20:
            score += 0.4
        elif word_count > 15:
            score += 0.3
        elif word_count > 10:
            score += 0.2

        # Сложные слова (длинные, с дефисами и т.д.)
        complex_words = sum(1 for w in words if len(w.word)
                            > 8 or '-' in w.word)
        if complex_words > 0:
            score += min(complex_words / word_count * 0.4, 0.4)

        # Слова-паразиты
        filler_count = sum(1 for w in words if w.is_filler)
        if filler_count > 0:
            score += min(filler_count * 0.1, 0.2)

        return min(score, 1.0)

    def _calculate_clarity_score(self, words: List[AdvancedWordTiming],
                                 avg_pause_duration: float) -> float:
        """Рассчитывает оценку ясности"""
        if not words:
            return 0.0

        score = 0.5

        # Паузы улучшают ясность
        if 0.3 <= avg_pause_duration <= 1.0:
            score += 0.3
        elif avg_pause_duration > 1.0:
            score -= 0.2

        # Длина слов
        avg_word_length = sum(len(w.word) for w in words) / len(words)
        if avg_word_length < 6:
            score += 0.1
        elif avg_word_length > 8:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _create_empty_timed_result(self, transcript: Transcript) -> TimedAnalysisResult:
        """Создает пустой результат с таймингами"""
        return TimedAnalysisResult(
            duration_sec=0,
            speaking_time_sec=0,
            speaking_ratio=0,
            words_total=0,
            words_per_minute=0,
            filler_words={
                "total": 0,
                "per_100_words": 0.0,
                "items": {},
                "distribution": {}
            },
            pauses={
                "total": 0,
                "avg_sec": 0,
                "max_sec": 0,
                "problematic_count": 0,
                "distribution": {}
            },
            phrases={
                "total": 0,
                "avg_words": 0,
                "avg_duration_sec": 0,
                "complexity_score": 0,
                "rhythm_score": 0
            },
            advice=[],
            transcript=transcript.text,
            timeline=SpeechTimeline(),
            speech_activity=[],
            speech_rate_windows=[],
            intensity_profile=[],
            emotion_timeline=[],
            gigachat_analysis=None,
            metadata={
                "analysis_version": "2.0",
                "includes_timings": False,
                "includes_visualization": False,
                "timestamp": None,
                "processing_time_sec": None
            }
        )
