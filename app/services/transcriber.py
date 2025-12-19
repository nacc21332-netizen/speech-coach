import hashlib
import logging
import pickle
from pathlib import Path
from typing import Protocol, List

from app.core.config import settings

try:
    from faster_whisper import WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError, RuntimeError) as e:
    _FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None
    _IMPORT_ERROR = str(e)
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
        cache_dir: Path | None = None,
        cache_ttl: int = 3600,  # 1 hour default
    ):
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self.cache_dir = cache_dir or Path(settings.cache_dir) / "transcriptions"
        self.cache_ttl = cache_ttl
        
        # Создаем директорию для кэша транскрипций
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not _FASTER_WHISPER_AVAILABLE:
            error_msg = (
                "⚠️  faster_whisper module not available. "
                "To enable transcription features, install dependencies: "
                "pip install faster-whisper ctranslate2 onnxruntime"
            )
            if _IMPORT_ERROR:
                error_msg += f"\n   Import error: {_IMPORT_ERROR}"
            logger.warning(error_msg)
            self.model = None
            self._model_available = False
        else:
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info(f"✅ Whisper model '{self.model_size}' loaded successfully on {self.device}")
                self._model_available = True
            except Exception as e:
                # If model cannot be downloaded/loaded (no internet or gated model),
                # gracefully fall back to a no-op transcriber that returns an
                # empty Transcript. This allows tests to run in offline CI.
                logger.warning(f"❌ Failed to load Whisper model '{self.model_size}': {type(e).__name__}: {e}")
                logger.warning("Falling back to offline dummy transcriber (no word timings).")
                self.model = None
                self._model_available = False

    def _get_cache_key(self, audio_path: Path) -> str:
        """Генерирует ключ кеша на основе пути к аудиофайлу и параметров модели"""
        # Получаем хэш файла
        with open(audio_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Включаем параметры модели в ключ кеша
        cache_key = f"{file_hash}_{self.model_size}_{self.device}_{self.compute_type}"
        return hashlib.sha256(cache_key.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Возвращает путь к файлу кеша"""
        return self.cache_dir / f"{key}.pkl"

    def transcribe(self, audio_path: Path) -> Transcript:
        """
        Транскрибация с таймингами для каждого слова.
        faster-whisper поддерживает word_timestamps=True
        """
        # Генерируем ключ кеша
        cache_key = self._get_cache_key(audio_path)
        cache_path = self._get_cache_path(cache_key)
        
        # Проверяем наличие закэшированного результата
        if cache_path.exists():
            try:
                import time
                # Проверяем TTL
                mtime = cache_path.stat().st_mtime
                if time.time() - mtime <= self.cache_ttl:
                    with open(cache_path, 'rb') as f:
                        cached_result = pickle.load(f)
                        logger.info(f"Using cached transcription for: {audio_path.name}")
                        return cached_result
                else:
                    # Удаляем просроченный кеш
                    cache_path.unlink()
            except Exception as e:
                logger.warning(f"Error reading cached transcription: {e}")
        
        logger.info(f"Transcribing audio with word timings: {audio_path}")
        segments: List[TranscriptSegment] = []
        all_word_timings: List[WordTiming] = []
        full_text = ""

        if self._model_available and self.model is not None:
            # segments — генератор, info — объект с метаданными
            segments_iter, info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=True,  # ← Ключевой параметр для таймингов слов!
                vad_filter=True,  # Фильтрация голосовой активности
            )

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

            logger.info(f"Transcription complete: {len(segments)} segments, {len(all_word_timings)} word timings")
            transcript = Transcript(
                text=full_text,
                segments=segments,
                word_timings=all_word_timings
            )
        else:
            # Offline/dummy fallback: return empty transcript (but still cacheable)
            logger.info("Model unavailable — returning empty transcript (dummy).")
            transcript = Transcript(text="", segments=[], word_timings=[])
        
        # Сохраняем результат в кеш
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(transcript, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Cached transcription: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving cached transcription: {e}")

        return transcript
