import os
import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

from app.services.audio_extractor import AudioExtractor
from app.services.transcriber import Transcriber
from app.services.analyzer import SpeechAnalyzer
from app.models.timed_analysis import TimedAnalysisResult


class SpeechAnalysisPipeline:
    """
    Координирует анализ речи из видео.
    """

    def __init__(
        self,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        analyzer: SpeechAnalyzer,
    ):
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.analyzer = analyzer

    async def analyze(self, file: UploadFile) -> TimedAnalysisResult:
        """
        Основной метод анализа видео.
        Возвращает полный результат с временными метками.
        """
        temp_video_path, temp_audio_path = await self._prepare_files(file)

        try:
            # 1. Извлекаем аудио
            self.audio_extractor.extract(temp_video_path, temp_audio_path)

            # 2. Транскрибируем с таймингами слов
            transcript = self._transcribe_with_timings(temp_audio_path)

            # 3. Анализируем
            result = self.analyzer.analyze(
                transcript, audio_path=temp_audio_path)
            return result

        finally:
            self._cleanup_files(temp_video_path, temp_audio_path)

    def _transcribe_with_timings(self, audio_path: Path):
        """
        Транскрибирует аудио с таймингами слов.
        Использует word_timestamps если доступно.
        """
        if hasattr(self.transcriber, 'transcribe_with_word_timings'):
            return self.transcriber.transcribe_with_word_timings(audio_path)
        else:
            # Резервный вариант
            return self.transcriber.transcribe(audio_path)

    async def _prepare_files(self, file: UploadFile) -> tuple[Path, Path]:
        """Подготавливает временные файлы."""
        suffix = Path(file.filename or "video").suffix or ".mp4"

        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_video_path = Path(tmp_video.name)
        tmp_video.close()

        await self._save_upload_to_path(file, temp_video_path)
        temp_audio_path = temp_video_path.with_suffix(".wav")

        return temp_video_path, temp_audio_path

    @staticmethod
    async def _save_upload_to_path(upload: UploadFile, dst: Path) -> None:
        """Сохраняет загруженный файл."""
        upload.file.seek(0)
        with dst.open("wb") as out_file:
            shutil.copyfileobj(upload.file, out_file)
        await upload.close()

    @staticmethod
    def _cleanup_files(*paths: Path) -> None:
        """Удаляет временные файлы."""
        for path in paths:
            try:
                if path.exists():
                    os.remove(path)
            except OSError:
                pass
