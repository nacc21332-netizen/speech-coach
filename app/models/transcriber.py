from pathlib import Path
from typing import Protocol, List

from faster_whisper import WhisperModel

from app.core.config import settings
from app.models.transcript import Transcript, TranscriptSegment, WordTiming


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
        self.model = WhisperModel(
            model_size or settings.whisper_model,
            device=device or settings.whisper_device,
            compute_type=compute_type or settings.whisper_compute_type,
        )

    def transcribe(self, audio_path: Path) -> Transcript:
        # Стандартная транскрипция (без таймингов слов)
        return self._transcribe_basic(audio_path)

    def transcribe_with_word_timings(self, audio_path: Path) -> Transcript:
        """
        Транскрибация с таймингами для каждого слова.
        faster-whisper поддерживает word_timestamps=True
        """
        # segments — генератор, info — объект с метаданными
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=True,  # ← Ключевой параметр!
            vad_filter=True
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

        return Transcript(
            text=full_text,
            segments=segments,
            word_timings=all_word_timings
        )

    def _transcribe_basic(self, audio_path: Path) -> Transcript:
        """
        Базовая транскрипция без таймингов слов (для обратной совместимости).
        """
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
        )

        segments: List[TranscriptSegment] = []
        texts: List[str] = []

        for seg in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text,
                )
            )
            texts.append(seg.text)

        full_text = " ".join(texts).strip()

        return Transcript(text=full_text, segments=segments)
