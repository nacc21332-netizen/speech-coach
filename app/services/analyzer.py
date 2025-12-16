from typing import List, Dict, Tuple, Any, Literal, Optional
from app.core.config import settings
import logging
from app.services import vad
import re
import wave
import struct
import math
from pathlib import Path
from pydantic import BaseModel
from app.services.gigachat import GigaChatClient
from app.services.cache import AnalysisCache

from app.models.transcript import Transcript, TranscriptSegment, WordTiming
from app.models.timed_analysis import (
    TimedFillerWord,
    TimedPause,
    SpeechRateWindow,
    TimedAnalysisData,
)
from app.models.analysis import (
    AnalysisResult, FillerWordsStats, PausesStats,
    PhraseStats, AdviceItem
)

# --------------------
# Константы "нормы"
# --------------------

MIN_PAUSE_GAP_SEC = settings.min_pause_gap_sec
MIN_COMFORT_WPM = 100.0
MAX_COMFORT_WPM = 180.0
LONG_PAUSE_SEC = settings.long_pause_sec
SILENCE_FACTOR = settings.silence_factor
PAUSE_SEGMENT_TIME_TOLERANCE = settings.pause_segment_time_tolerance
SPEECH_RATE_WINDOW_SIZE = 30.0  # секунд
SPEECH_RATE_WINDOW_STEP = 15.0  # секунд

# --------------------
# Слова-паразиты
# --------------------

FILLER_DEFINITIONS: List[Tuple[str, str]] = [
    # Русские звучания
    ("э-э", r"\bэ+([- ]э+)*\b"),
    ("эм", r"\bэм+\b"),
    ("мм", r"\bмм+\b"),
    ("ну", r"\bну\b"),
    ("вот", r"\bвот\b"),
    ("там", r"\bтам\b"),
    ("да", r"\bда\b"),
    ("короче", r"\bкороче\b"),
    ("вроде", r"\bвроде\b"),
    ("вроде бы", r"\bвроде бы\b"),
    ("в общем", r"\bв общем\b"),
    ("кстати", r"\bкстати\b"),
    ("собственно", r"\bсобственно\b"),
    # Русские конструкции
    ("как бы", r"\bкак бы\b"),
    ("типа", r"\bтипа\b"),
    ("то есть", r"\bто есть\b"),
    ("значит", r"\bзначит\b"),
    ("получается", r"\bполучается\b"),

    # Английские звучания
    ("uh", r"\buh+\b"),
    ("um", r"\bum+\b"),
    ("er", r"\ber+\b"),

    # Английские конструкции
    ("like", r"\blike\b"),
    ("so", r"\bso\b"),
    ("you know", r"\byou know\b"),
    ("i mean", r"\bi mean\b"),
]

COMPILED_FILLERS: List[Tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE))
    for name, pattern in FILLER_DEFINITIONS
]


class EnhancedAnalysisResult(BaseModel):
    """Расширенный результат анализа с таймингами"""
    # Основные метрики из AnalysisResult
    duration_sec: float
    speaking_time_sec: float
    speaking_ratio: float
    words_total: int
    words_per_minute: float
    filler_words: FillerWordsStats
    pauses: PausesStats
    phrases: PhraseStats
    advice: List[AdviceItem]
    transcript: str
    gigachat_analysis: Optional[Any] = None

    # Детализированные данные с таймингами
    timed_data: TimedAnalysisData = TimedAnalysisData()


class SpeechAnalyzer:
    async def analyze(
        self,
        transcript: Transcript,
        audio_path: Path | None = None,
        include_timings: bool = True,
        gigachat_client: Optional[GigaChatClient] = None,
        cache: Optional[AnalysisCache] = None,
    ) -> EnhancedAnalysisResult:
        """
        Анализирует речь с возможностью включения таймингов.
        """
        # Базовый анализ
        base_result = self._analyze_basic(transcript, audio_path)

        # Дополнительный анализ с таймингами
        timed_data = TimedAnalysisData()

        if include_timings and transcript.word_timings:
            timed_data = await self._analyze_with_timings(transcript, audio_path, gigachat_client, cache)

        return EnhancedAnalysisResult(
            **base_result.dict(),
            timed_data=timed_data
        )

    def _analyze_basic(
        self,
        transcript: Transcript,
        audio_path: Path | None = None,
    ) -> AnalysisResult:
        """Базовый анализ без таймингов"""
        segments = transcript.segments
        if not segments:
            return self._empty_analysis_result(transcript.text or "")

        duration_sec = float(segments[-1].end)
        full_text = transcript.text or ""
        words = self._split_words(full_text)
        words_total = len(words)

        # Расчет времени говорения и пауз
        speaking_time_sec, pauses_raw = self._calculate_speaking_stats(
            segments)

        # Фильтрация пауз по аудио
        pauses_filtered = self._filter_pauses(audio_path, pauses_raw, segments)

        # Основные метрики
        words_per_minute = self._calculate_wpm(words_total, speaking_time_sec)
        speaking_ratio = speaking_time_sec / duration_sec if duration_sec > 0 else 0.0

        # Статистика
        filler_total, filler_detail = self._count_fillers(full_text)
        pauses_stats = self._summarize_pauses(pauses_filtered)
        phrase_stats = self._build_phrase_stats(segments, pauses_filtered)
        advice = self._generate_advice(
            words_per_minute, filler_total, words_total, pauses_stats, phrase_stats)

        # Статистика слов-паразитов
        # Вычисляем слова-паразиты на 100 слов ТОЧНО из полного текста
        filler_stats = FillerWordsStats(
            total=filler_total,
            per_100_words=round(
                (filler_total / words_total * 100), 2
            ) if words_total > 0 else 0.0,
            items=[
                {"word": name, "count": filler_detail.get(name, 0)}
                for name, _ in COMPILED_FILLERS
                if filler_detail.get(name, 0) > 0
            ],
        )

        return AnalysisResult(
            duration_sec=round(duration_sec, 2),
            speaking_time_sec=round(speaking_time_sec, 2),
            speaking_ratio=round(speaking_ratio, 3),
            words_total=words_total,
            words_per_minute=round(words_per_minute, 1),

            filler_words=filler_stats,
            pauses=pauses_stats,
            phrases=phrase_stats,

            advice=advice,
            transcript=full_text,
            gigachat_analysis=None,
        )

    async def _analyze_with_timings(self, transcript: Transcript, audio_path: Path | None = None, gigachat_client: Optional[GigaChatClient] = None, cache: Optional[AnalysisCache] = None) -> TimedAnalysisData:
        """Анализ с использованием таймингов слов"""
        if not transcript.word_timings:
            return TimedAnalysisData()

        return TimedAnalysisData(
            filler_words_detailed=await self._find_fillers_with_exact_timings(
                transcript, gigachat_client, cache),
            pauses_detailed=self._analyze_pauses_with_word_timings(transcript, audio_path),
            speech_rate_windows=self._calculate_speech_windows_by_words(
                transcript),
            word_timings_count=len(transcript.word_timings),
            speaking_activity=self._build_speaking_activity(transcript),
        )

    async def _find_fillers_with_exact_timings(self, transcript: Transcript, gigachat_client: Optional[GigaChatClient] = None, cache: Optional[AnalysisCache] = None) -> List[TimedFillerWord]:
        """Находит слова-паразиты с точными таймингами, используя контекстный анализ при наличии GigaChat клиента"""
        if gigachat_client is not None and settings.llm_fillers_enabled:
            # Используем контекстный анализ с GigaChat
            from app.services.contextual_filler_analyzer import ContextualFillerAnalyzer
            contextual_analyzer = ContextualFillerAnalyzer(gigachat_client, cache)
            return await contextual_analyzer.analyze_fillers_with_context(transcript)
        else:
            # Используем базовый анализ без контекста
            fillers = []
            for word_timing in transcript.word_timings:
                word_text = word_timing.word.lower().strip().strip(",.!?;:()\"'")
                # Нормализуем повторяющиеся символы (например, 'ээээ' -> 'ээ')
                word_text_norm = re.sub(r"(.)\1{2,}", r"\1\1", word_text)
                word_text_norm = word_text_norm.replace("-", " ")

                for filler_name, pattern in COMPILED_FILLERS:
                    # Используем search на нормализованной версии
                    if pattern.search(word_text) or pattern.search(word_text_norm) or (filler_name in word_text_norm):
                        fillers.append(TimedFillerWord(
                            word=filler_name,
                            timestamp=round(word_timing.start, 3),
                            exact_word=word_timing.word,
                            confidence=word_timing.confidence,
                            duration=round(max(0.0, word_timing.end - word_timing.start), 3)
                        ))
                        break

            return fillers

    def _analyze_pauses_with_word_timings(self, transcript: Transcript, audio_path: Path | None = None) -> List[TimedPause]:
        """Анализирует паузы между словами"""
        # Note: audio_path optional will be applied in filtering stage
        if len(transcript.word_timings) < 2:
            return []

        raw_pauses = []
        for i in range(len(transcript.word_timings) - 1):
            current_word = transcript.word_timings[i]
            next_word = transcript.word_timings[i + 1]

            pause_duration = next_word.start - current_word.end

            if pause_duration >= MIN_PAUSE_GAP_SEC:
                pause_type = self._classify_pause(pause_duration)
                raw_pauses.append({
                    "start": round(current_word.end, 3),
                    "end": round(next_word.start, 3),
                    "duration": round(pause_duration, 3),
                    "type": pause_type,
                    "before_word": current_word.word,
                    "after_word": next_word.word
                })

        # Apply audio-based filtering to discard false pauses caused by mis-segmentation
        filtered_raw = raw_pauses
        if audio_path is not None and raw_pauses:
            try:
                filtered_raw = self._filter_noisy_pauses(audio_path, raw_pauses, transcript.segments)
            except Exception:
                filtered_raw = raw_pauses

        pauses = [TimedPause(
            start=p["start"],
            end=p["end"],
            duration=p["duration"],
            type=p.get("type", self._classify_pause(p["duration"])),
            before_word=p.get("before_word"),
            after_word=p.get("after_word")
        ) for p in filtered_raw]

        return pauses

    def _calculate_speech_windows_by_words(self, transcript: Transcript) -> List[SpeechRateWindow]:
        """Рассчитывает темп речи в скользящих окнах на основе таймингов слов"""
        if not transcript.word_timings:
            return []

        total_duration = max(wt.end for wt in transcript.word_timings)
        windows = []
        current_start = 0.0

        while current_start < total_duration:
            window_end = min(
                current_start + SPEECH_RATE_WINDOW_SIZE, total_duration)

            # Считаем слова в окне
            words_in_window = 0
            speaking_time = 0.0

            for word_timing in transcript.word_timings:
                if word_timing.start >= current_start and word_timing.end <= window_end:
                    words_in_window += 1
                    speaking_time += (word_timing.end - word_timing.start)
                elif word_timing.start < window_end and word_timing.end > current_start:
                    # Частичное пересечение
                    overlap_start = max(word_timing.start, current_start)
                    overlap_end = min(word_timing.end, window_end)
                    if overlap_end > overlap_start:
                        words_in_window += (overlap_end - overlap_start) / \
                            (word_timing.end - word_timing.start)
                        speaking_time += (overlap_end - overlap_start)

            wpm = words_in_window / \
                (speaking_time / 60.0) if speaking_time > 0 else 0.0

            windows.append(SpeechRateWindow(
                window_start=round(current_start, 2),
                window_end=round(window_end, 2),
                word_count=int(words_in_window),
                words_per_minute=round(wpm, 1),
                speaking_time=round(speaking_time, 2)
            ))

            current_start += SPEECH_RATE_WINDOW_STEP

        return windows

    def _build_speaking_activity(self, transcript: Transcript, resolution: float = 0.5) -> List[Dict[str, float]]:
        """Создает массив активности говорения для визуализации"""
        if not transcript.word_timings:
            return []

        total_duration = max(wt.end for wt in transcript.word_timings)
        activity = []
        current_time = 0.0

        while current_time <= total_duration:
            is_speaking = 0.0

            for word_timing in transcript.word_timings:
                if word_timing.start <= current_time <= word_timing.end:
                    is_speaking = 1.0
                    break

            activity.append({
                "time": round(current_time, 2),
                "is_speaking": is_speaking
            })

            current_time += resolution

        return activity

    @staticmethod
    def _classify_pause(duration: float) -> str:
        """Классифицирует паузу по длительности."""
        if duration < 1.0:
            return "natural"
        elif duration < 2.5:
            return "dramatic"
        elif duration < 5.0:
            return "long"
        else:
            return "awkward"

    # Остальные методы остаются без изменений (из предыдущей версии)
    def _empty_analysis_result(self, text: str) -> AnalysisResult:
        """Возвращает результат для пустого транскрипта"""
        return AnalysisResult(
            duration_sec=0.0,
            speaking_time_sec=0.0,
            speaking_ratio=0.0,
            words_total=0,
            words_per_minute=0.0,
            filler_words=FillerWordsStats(
                total=0,
                per_100_words=0.0,
                items=[]
            ),
            pauses=PausesStats(
                count=0,
                avg_sec=0.0,
                max_sec=0.0,
                long_pauses=[]
            ),
            phrases=PhraseStats(
                count=0,
                avg_words=0.0,
                avg_duration_sec=0.0,
                min_words=0,
                max_words=0,
                min_duration_sec=0.0,
                max_duration_sec=0.0,
                length_classification="insufficient_data",
                rhythm_variation="insufficient_data"
            ),
            advice=[
                AdviceItem(
                    category="speech_rate",
                    severity="warning",
                    title="Недостаточно данных",
                    observation="В аудио не обнаружено речи или транскрипт пуст",
                    recommendation="Попробуйте записать более четкую речь или увеличьте громкость"
                )
            ],
            transcript=text,
            gigachat_analysis=None,
        )

    @staticmethod
    def _calculate_speaking_stats(segments: List[TranscriptSegment]) -> Tuple[float, List[Dict[str, float]]]:
        """Рассчитывает время говорения и паузы"""
        speaking_time_sec = 0.0
        pauses_raw = []
        last_end = None

        for seg in segments:
            start = float(seg.start)
            end = float(seg.end)
            seg_duration = max(0.0, end - start)
            speaking_time_sec += seg_duration

            if last_end is not None:
                gap = start - last_end
                if gap >= MIN_PAUSE_GAP_SEC:
                    pauses_raw.append({
                        "start": last_end,
                        "end": start,
                        "duration": gap
                    })
            last_end = end

        # Debug: log raw pauses count and some examples
        try:
            logger = logging.getLogger(__name__)
            logger.debug("_calculate_speaking_stats: speaking_time_sec=%s, raw_pauses_count=%d", speaking_time_sec, len(pauses_raw))
            if pauses_raw:
                # Log first 5 pauses for brevity
                logger.debug("_calculate_speaking_stats: raw_pauses_sample=%s", pauses_raw[:5])
        except Exception:
            pass

        return speaking_time_sec, pauses_raw

    def _filter_pauses(
        self,
        audio_path: Path | None,
        pauses: List[Dict[str, float]],
        segments: List[TranscriptSegment]
    ) -> List[Dict[str, float]]:
        """Фильтрует паузы по аудио"""
        if audio_path is None or not pauses:
            return pauses

        try:
            return self._filter_noisy_pauses(audio_path, pauses, segments)
        except Exception as e:
            # При ошибке чтения аудио возвращаем сырые паузы
            return pauses

    @staticmethod
    def _calculate_wpm(words_total: int, speaking_time_sec: float) -> float:
        """Рассчитывает слова в минуту"""
        if speaking_time_sec <= 0 or words_total <= 0:
            return 0.0
        return words_total / (speaking_time_sec / 60.0)

    @staticmethod
    def _split_words(text: str) -> List[str]:
        """Разделяет текст на слова"""
        if not text:
            return []
        return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())

    @staticmethod
    def _count_fillers(text: str) -> Tuple[int, Dict[str, int]]:
        """Считает слова-паразиты"""
        if not text:
            return 0, {}

        counts: Dict[str, int] = {}
        total = 0

        # Нормализуем текст: уменьшение повторений символов и приведение к нижнему регистру
        normalized_text = re.sub(r"(.)\1{2,}", r"\1\1", text.lower())
        normalized_text = normalized_text.replace('-', ' ')

        # Используем finditer вместо findall чтобы правильно считать совпадения
        found_positions = set()

        for name, pattern in COMPILED_FILLERS:
            for match in pattern.finditer(normalized_text):
                start, end = match.span()
                # Проверяем, не пересекается ли это совпадение с другими
                if not any(start < pos <= end or pos >= start and pos < end 
                          for pos in found_positions):
                    found_positions.add(start)
                    counts[name] = counts.get(name, 0) + 1
                    total += 1

        return total, counts

    @staticmethod
    def _filter_noisy_pauses(
        audio_path: Path,
        pauses: List[Dict[str, float]],
        segments: List[TranscriptSegment],
    ) -> List[Dict[str, float]]:
        """Фильтрует шумные паузы"""
        wav_read_failed = False
        framerate = None
        frames = b''
        try:
            with wave.open(str(audio_path), "rb") as wf:
                n_channels, sampwidth, framerate, n_frames, *_ = wf.getparams()

                if n_channels != 1 or sampwidth != 2:
                    wav_read_failed = True
                else:
                    frames = wf.readframes(n_frames)
        except Exception:
            wav_read_failed = True

        num_samples = len(frames) // 2 if frames else 0
        logger = logging.getLogger(__name__)
        logger.debug("_filter_noisy_pauses: wav_read_failed=%s, framerate=%s, num_samples=%s", wav_read_failed, framerate, num_samples)
        # Try early VAD detection — if we failed to read WAV, use VAD results to filter pauses immediately.
        try:
            early_vad_segments = vad.detect_speech_regions(audio_path, settings.use_pyannote_vad, settings.use_webrtc_vad, webrtc_mode=settings.webrtc_vad_mode, pyannote_model=settings.pyannote_model)
        except Exception:
            early_vad_segments = []

        logger.debug("_filter_noisy_pauses: early_vad_segments_count=%s sample=%s", len(early_vad_segments), early_vad_segments[:3])

        if wav_read_failed and early_vad_segments:
            def has_vad_activity_early(start_s: float, end_s: float) -> bool:
                for vstart, vend in early_vad_segments:
                    if not (vend <= start_s or vstart >= end_s):
                        return True
                return False

            filtered = []
            for pause in pauses:
                start_s = pause["start"]
                end_s = pause["end"]
                if end_s <= start_s:
                    continue
                if not has_vad_activity_early(start_s, end_s):
                    filtered.append(pause)
            logger.debug("_filter_noisy_pauses: filtered_by_early_vad_count=%s", len(filtered))
            return filtered

        samples = struct.unpack("<{}h".format(num_samples), frames)

        def segment_rms(start_idx: int, end_idx: int) -> float:
            """Вычисляет RMS для сегмента"""
            count = end_idx - start_idx
            if count <= 0:
                return 0.0

            sum_sq = sum(samples[i] * samples[i]
                         for i in range(start_idx, end_idx))
            return math.sqrt(sum_sq / count)

        # RMS речевых сегментов
        speech_rms_values = []
        for seg in segments:
            start_s = float(seg.start)
            end_s = float(seg.end)
            start_idx = max(0, int(start_s * framerate))
            end_idx = min(num_samples, int(end_s * framerate))

            if end_idx - start_idx < int(0.2 * framerate):
                continue

            r = segment_rms(start_idx, end_idx)
            if r > 0:
                speech_rms_values.append(r)

        # Если не получилось вычислить RMS на основе сегментов (например, сегменты тихие),
        # делаем запасной подсчёт медианы RMS по всему файлу в скользящих окнах.
        if not speech_rms_values:
            window_size = int(0.1 * framerate) if framerate else 0
            if window_size > 0:
                for i in range(0, num_samples, window_size):
                    r = segment_rms(i, min(num_samples, i + window_size))
                    if r > 0:
                        speech_rms_values.append(r)

        # Если после всех попыток RMS не получился, попробуем фильтровать паузы только по VAD (если он доступен).
        if not speech_rms_values:
            try:
                vad_segments = vad.detect_speech_regions(audio_path, settings.use_pyannote_vad, settings.use_webrtc_vad, webrtc_mode=settings.webrtc_vad_mode, pyannote_model=settings.pyannote_model)
            except Exception:
                vad_segments = []

            def has_vad_activity_local(start_s: float, end_s: float) -> bool:
                for vstart, vend in vad_segments:
                    if not (vend <= start_s or vstart >= end_s):
                        return True
                return False

            logger.debug("_filter_noisy_pauses: fallback vad_segments_count=%s sample=%s", len(vad_segments), vad_segments[:3])
            if vad_segments:
                filtered = []
                for pause in pauses:
                    start_s = pause["start"]
                    end_s = pause["end"]
                    if end_s <= start_s:
                        continue
                    if not has_vad_activity_local(start_s, end_s):
                        filtered.append(pause)
                logger.debug("_filter_noisy_pauses: filtered_by_vad_only_count=%s", len(filtered))
                return filtered

            return pauses

        # Медиана громкости речи
        speech_rms_values.sort()
        median_speech_rms = speech_rms_values[len(speech_rms_values) // 2]
        silence_threshold = median_speech_rms * SILENCE_FACTOR
        logger.debug("_filter_noisy_pauses: median_speech_rms=%s, silence_threshold=%s", median_speech_rms, silence_threshold)

        # Пытаемся использовать VAD (pyannote / webrtcvad) для исключения ложных пауз
        try:
            vad_segments = vad.detect_speech_regions(audio_path, settings.use_pyannote_vad, settings.use_webrtc_vad, webrtc_mode=settings.webrtc_vad_mode, pyannote_model=settings.pyannote_model)
        except Exception as e:
            vad_segments = []
        logger.debug("_filter_noisy_pauses: vad_segments_count=%s sample=%s", len(vad_segments), vad_segments[:3])

        def has_vad_activity(start_s: float, end_s: float) -> bool:
            for vstart, vend in vad_segments:
                # If VAD speech overlaps with the pause
                if not (vend <= start_s or vstart >= end_s):
                    # Overlap exists
                    return True
            return False

        # Если не удалось прочитать форму (wav) — пробуем фильтровать только по VAD (если доступен)
        if wav_read_failed and vad_segments:
            filtered = []
            for pause in pauses:
                start_s = pause["start"]
                end_s = pause["end"]
                if end_s <= start_s:
                    continue
                if not has_vad_activity(start_s, end_s):
                    filtered.append(pause)
            return filtered

        # Фильтрация пауз по RMS и VAD
        filtered = []
        for pause in pauses:
            start_s = pause["start"]
            end_s = pause["end"]

            if end_s <= start_s:
                continue

            start_idx = max(0, int(start_s * framerate))
            end_idx = min(num_samples, int(end_s * framerate))

            if end_idx <= start_idx:
                continue

            # If VAD was available and detects speech in this gap, skip it
            if vad_segments and has_vad_activity(start_s, end_s):
                continue

            r = segment_rms(start_idx, end_idx)

            if r < silence_threshold:
                filtered.append(pause)

        # Debug: log filtering summary
        logger.debug("_filter_noisy_pauses: raw_count=%s, filtered_count=%s", len(pauses), len(filtered))
        if filtered:
            logger.debug("_filter_noisy_pauses: filtered_sample=%s", filtered[:5])

        # For the first few pauses, log detailed decision info
        for i, pause in enumerate(pauses[:5]):
            start_s = pause["start"]
            end_s = pause["end"]
            start_idx = max(0, int(start_s * framerate))
            end_idx = min(num_samples, int(end_s * framerate))
            r = segment_rms(start_idx, end_idx) if end_idx > start_idx else 0.0
            vad_active = bool(vad_segments and has_vad_activity(start_s, end_s))
            kept = pause in filtered
            logger.debug("_filter_noisy_pauses: pause_idx=%s start=%s end=%s dur=%s rms=%s vad_active=%s kept=%s", i, start_s, end_s, pause.get("duration"), r, vad_active, kept)

        return filtered

    @staticmethod
    def _summarize_pauses(pauses: List[Dict[str, float]]) -> PausesStats:
        """Суммирует статистику пауз"""
        if not pauses:
            return PausesStats(
                count=0,
                avg_sec=0.0,
                max_sec=0.0,
                long_pauses=[],
            )

        durations = [p["duration"] for p in pauses]
        avg_sec = sum(durations) / len(pauses)
        max_sec = max(durations)

        # Длинные паузы
        long_pauses = [
            {
                "start": round(p["start"], 2),
                "end": round(p["end"], 2),
                "duration": round(p["duration"], 2),
            }
            for p in pauses
            if p["duration"] >= LONG_PAUSE_SEC
        ]

        # Топ-3 самых длинных
        long_pauses = sorted(
            long_pauses, key=lambda p: p["duration"], reverse=True)[:3]

        return PausesStats(
            count=len(pauses),
            avg_sec=round(avg_sec, 2),
            max_sec=round(max_sec, 2),
            long_pauses=long_pauses,
        )

    def _build_phrase_stats(
        self,
        segments: List[TranscriptSegment],
        pauses: List[Dict[str, float]],
    ) -> PhraseStats:
        """Строит статистику фраз"""
        if not segments:
            return self._empty_phrase_stats()

        # Определяем границы фраз
        boundary_after_idx = self._find_phrase_boundaries(segments, pauses)

        # Собираем фразы
        phrases_durations, phrases_words = self._collect_phrases(
            segments, boundary_after_idx)

        if not phrases_words:
            return self._empty_phrase_stats()

        return self._calculate_phrase_stats(phrases_durations, phrases_words)

    def _find_phrase_boundaries(
        self,
        segments: List[TranscriptSegment],
        pauses: List[Dict[str, float]]
    ) -> set:
        """Находит индексы сегментов, после которых есть паузы"""
        boundary_after_idx = set()

        for pause in pauses:
            pause_start = pause["start"]
            best_idx = None
            best_diff = float("inf")

            for i, seg in enumerate(segments):
                diff = abs(seg.end - pause_start)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx is not None and best_diff <= PAUSE_SEGMENT_TIME_TOLERANCE:
                boundary_after_idx.add(best_idx)

        return boundary_after_idx

    def _collect_phrases(
        self,
        segments: List[TranscriptSegment],
        boundary_after_idx: set
    ) -> Tuple[List[float], List[int]]:
        """Собирает фразы из сегментов"""
        phrases_durations: List[float] = []
        phrases_words: List[int] = []
        phrase_start_idx = 0

        for idx in range(len(segments)):
            is_boundary = idx in boundary_after_idx
            is_last_seg = idx == len(segments) - 1

            if is_boundary or is_last_seg:
                phrase_end_idx = idx
                segs = segments[phrase_start_idx: phrase_end_idx + 1]

                if segs:
                    dur, wcount = self._phrase_metrics(segs)
                    if wcount > 0 and dur > 0:
                        phrases_durations.append(dur)
                        phrases_words.append(wcount)

                phrase_start_idx = idx + 1

        return phrases_durations, phrases_words

    def _calculate_phrase_stats(
        self,
        phrases_durations: List[float],
        phrases_words: List[int]
    ) -> PhraseStats:
        """Рассчитывает статистику фраз"""
        count = len(phrases_words)
        avg_words = sum(phrases_words) / count
        avg_dur = sum(phrases_durations) / count

        # Классификация по длине фраз
        if avg_words < 8:
            length_class = "short_phrases"
        elif avg_words <= 25:
            length_class = "balanced"
        else:
            length_class = "long_phrases"

        # Вариативность ритма
        rhythm_var = self._calculate_rhythm_variation(
            phrases_durations, avg_dur, count)

        return PhraseStats(
            count=count,
            avg_words=round(avg_words, 1),
            avg_duration_sec=round(avg_dur, 2),
            min_words=min(phrases_words),
            max_words=max(phrases_words),
            min_duration_sec=round(min(phrases_durations), 2),
            max_duration_sec=round(max(phrases_durations), 2),
            length_classification=length_class,
            rhythm_variation=rhythm_var,
        )

    @staticmethod
    def _calculate_rhythm_variation(
        durations: List[float],
        mean_dur: float,
        count: int
    ) -> str:
        """Рассчитывает вариативность ритма"""
        if count < 2 or mean_dur <= 0:
            return "insufficient_data"

        variance = sum((d - mean_dur) ** 2 for d in durations) / count
        std = math.sqrt(variance)
        cv = std / mean_dur

        if cv < 0.25:
            return "uniform"
        elif cv < 0.6:
            return "moderately_variable"
        else:
            return "highly_variable"

    @staticmethod
    def _empty_phrase_stats() -> PhraseStats:
        """Возвращает пустую статистику фраз"""
        return PhraseStats(
            count=0,
            avg_words=0.0,
            avg_duration_sec=0.0,
            min_words=0,
            max_words=0,
            min_duration_sec=0.0,
            max_duration_sec=0.0,
            length_classification="insufficient_data",
            rhythm_variation="insufficient_data",
        )

    def _phrase_metrics(
        self,
        segs: List[TranscriptSegment],
    ) -> Tuple[float, int]:
        """Возвращает длительность и количество слов во фразе"""
        if not segs:
            return 0.0, 0

        start = float(segs[0].start)
        end = float(segs[-1].end)
        duration = max(0.0, end - start)

        text = " ".join(s.text for s in segs)
        words = self._split_words(text)

        return duration, len(words)

    # -------------------------------------------------
    # Советы
    # -------------------------------------------------

    @staticmethod
    def _generate_advice(
        words_per_minute: float,
        filler_total: int,
        words_total: int,
        pauses_stats: PausesStats,
        phrase_stats: PhraseStats,
    ) -> List[AdviceItem]:
        """Генерирует рекомендации"""
        advice = []

        # 1. Темп речи
        advice.append(SpeechAnalyzer._generate_speech_rate_advice(
            words_per_minute, words_total))

        # 2. Слова-паразиты
        advice.append(SpeechAnalyzer._generate_filler_advice(
            filler_total, words_total))

        # 3. Паузы
        advice.append(SpeechAnalyzer._generate_pauses_advice(pauses_stats))

        # 4. Структура фраз
        advice.append(SpeechAnalyzer._generate_phrasing_advice(phrase_stats))

        return advice

    @staticmethod
    def _generate_speech_rate_advice(wpm: float, words_total: int) -> AdviceItem:
        """Генерирует рекомендации по темпу речи"""
        if words_total == 0 or wpm == 0:
            return AdviceItem(
                category="speech_rate",
                severity="info",
                title="Темп речи",
                observation="Не удалось определить темп речи",
                recommendation="Запишите более продолжительный фрагмент с отчётливой речью"
            )

        if wpm < MIN_COMFORT_WPM:
            return AdviceItem(
                category="speech_rate",
                severity="suggestion",
                title="Темп речи",
                observation=f"Темп речи составляет примерно {wpm:.1f} слов в минуту, что ниже типичного диапазона ({
                    MIN_COMFORT_WPM:.0f}–{MAX_COMFORT_WPM:.0f} слов в минуту)",
                recommendation="Попробуйте немного ускорить речь, сокращая избыточные паузы"
            )
        elif wpm > MAX_COMFORT_WPM:
            return AdviceItem(
                category="speech_rate",
                severity="suggestion",
                title="Темп речи",
                observation=f"Темп речи составляет примерно {wpm:.1f} слов в минуту, что выше типичного диапазона ({
                    MIN_COMFORT_WPM:.0f}–{MAX_COMFORT_WPM:.0f} слов в минуту)",
                recommendation="Попробуйте замедлить речь, делая более заметные паузы между фразами"
            )
        else:
            return AdviceItem(
                category="speech_rate",
                severity="info",
                title="Темп речи",
                observation=f"Темп речи составляет примерно {
                    wpm:.1f} слов в минуту, что находится в пределах нормы",
                recommendation="Сохраняйте выбранный темп и варьируйте его для выделения ключевых мыслей"
            )

    @staticmethod
    def _generate_filler_advice(filler_total: int, words_total: int) -> AdviceItem:
        """Генерирует рекомендации по словам-паразитам"""
        fillers_per_100 = (filler_total / words_total *
                           100) if words_total else 0.0

        if filler_total == 0:
            return AdviceItem(
                category="filler_words",
                severity="info",
                title="Слова-паразиты",
                observation="Слов-паразитов не обнаружено",
                recommendation="Продолжайте контролировать свою речь"
            )
        elif fillers_per_100 <= 3:
            return AdviceItem(
                category="filler_words",
                severity="info",
                title="Слова-паразиты",
                observation=f"Слова-паразиты присутствуют в небольшом количестве ({
                    fillers_per_100:.1f} на 100 слов)",
                recommendation="Старайтесь заменять слова-паразиты короткими паузами"
            )
        elif fillers_per_100 <= 8:
            return AdviceItem(
                category="filler_words",
                severity="suggestion",
                title="Слова-паразиты",
                observation=f"Заметное количество слов-паразитов ({
                    fillers_per_100:.1f} на 100 слов)",
                recommendation="Обратите внимание на наиболее частые слова-паразиты и сознательно избегайте их"
            )
        else:
            return AdviceItem(
                category="filler_words",
                severity="warning",
                title="Слова-паразиты",
                observation=f"Высокий уровень слов-паразитов ({
                    fillers_per_100:.1f} на 100 слов)",
                recommendation="Потренируйтесь говорить, сознательно избегая слов-паразитов, используйте паузы вместо них"
            )

    @staticmethod
    def _generate_pauses_advice(pauses_stats: PausesStats) -> AdviceItem:
        """Генерирует рекомендации по паузам"""
        if pauses_stats.count == 0:
            return AdviceItem(
                category="pauses",
                severity="info",
                title="Паузы в речи",
                observation="В речи практически отсутствуют паузы",
                recommendation="Используйте короткие паузы для структурирования речи"
            )

        long_count = len(pauses_stats.long_pauses)
        if long_count > 0 and long_count / pauses_stats.count > 0.3:
            return AdviceItem(
                category="pauses",
                severity="suggestion",
                title="Паузы в речи",
                observation=f"Обнаружены длинные паузы (до {
                    pauses_stats.max_sec:.1f} секунд)",
                recommendation="Сократите слишком длинные паузы или заполните их связующими фразами"
            )
        else:
            return AdviceItem(
                category="pauses",
                severity="info",
                title="Паузы в речи",
                observation=f"Паузы используются естественно (средняя длина {
                    pauses_stats.avg_sec:.1f} секунд)",
                recommendation="Продолжайте использовать паузы для лучшего восприятия речи"
            )

    @staticmethod
    def _generate_phrasing_advice(phrase_stats: PhraseStats) -> AdviceItem:
        """Генерирует рекомендации по структуре фраз"""
        if phrase_stats.count <= 1:
            return AdviceItem(
                category="phrasing",
                severity="info",
                title="Структура фраз",
                observation="Недостаточно данных для анализа структуры фраз",
                recommendation="Используйте паузы для разделения смысловых блоков"
            )

        observation = f"Средняя длина фразы: {
            phrase_stats.avg_words:.1f} слов ({phrase_stats.avg_duration_sec:.1f} секунд)"
        recommendation = ""
        severity = "info"

        if phrase_stats.length_classification == "short_phrases":
            observation += ". Фразы довольно короткие"
            recommendation = "Объединяйте связанные мысли в более длинные фразы"
            severity = "suggestion"
        elif phrase_stats.length_classification == "long_phrases":
            observation += ". Фразы довольно длинные"
            recommendation = "Разбивайте сложные мысли на более короткие фразы"
            severity = "suggestion"
        else:
            observation += ". Длина фраз сбалансирована"
            recommendation = "Продолжайте использовать фразы разной длины для поддержания ритма"

        return AdviceItem(
            category="phrasing",
            severity=severity,
            title="Структура фраз",
            observation=observation,
            recommendation=recommendation
        )
