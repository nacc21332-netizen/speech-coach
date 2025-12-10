import logging
from pathlib import Path
from typing import Protocol, List

from faster_whisper import WhisperModel

from app.core.config import settings
from app.models.transcript import Transcript, TranscriptSegment, WordTiming

logger = logging.getLogger(__name__)


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> Transcript:
        ...


class LocalWhisperTranscriber:
    """
    Использует локальную модель Whisper через faster-whisper.
    Модель скачивается при первом запуске (несколько сотен МБ).
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type

        logger.info(f"Loading Whisper model: {
                    self.model_size} on {self.device}")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info(f"Whisper model loaded successfully")

    def transcribe(self, audio_path: Path) -> Transcript:
        """
        Транскрибация с таймингами для каждого слова.
        faster-whisper поддерживает word_timestamps=True
        """
        logger.info(f"Transcribing audio with word timings: {audio_path}")

        # segments — генератор, info — объект с метаданными
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=True,  # ← Ключевой параметр для таймингов слов!
            vad_filter=True,  # Фильтрация голосовой активности
        )

        segments: List[TranscriptSegment] = []
        all_word_timings: List[WordTiming] = []
        texts: List[str] = []

        for seg in segments_iter:
            words_in_segment: List[WordTiming] = []

            # seg.words содержит список объектов с start, end, word
            if hasattr(seg, 'words') and seg.words:
                for word_info in seg.words:
                    word_timing = WordTiming(
                        word=word_info.word,
                        start=float(word_info.start),
                        end=float(word_info.end),
                        confidence=getattr(word_info, 'probability', None)
                    )
                    words_in_segment.append(word_timing)
                    all_word_timings.append(word_timing)

            segment = TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text,
                words=words_in_segment
            )
            segments.append(segment)
            texts.append(seg.text)

        full_text = " ".join(texts).strip()

        logger.info(f"Transcription complete: {len(segments)} segments, {
                    len(all_word_timings)} word timings")

        return Transcript(
            text=full_text,
            segments=segments,
            word_timings=all_word_timings
        )
