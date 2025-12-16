import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import logging
import time
import asyncio
import aiofiles

from fastapi import UploadFile

from app.services.audio_extractor_advanced import AdvancedFfmpegAudioExtractor, TimeoutException
from app.services.transcriber import Transcriber
from app.services.analyzer import SpeechAnalyzer, EnhancedAnalysisResult
from app.services.gigachat import GigaChatClient
from app.services.metrics_collector import MetricsCollector
from app.services.cache import AnalysisCache, cache_analysis
from app.models.analysis import AnalysisResult
from app.core.config import settings
from app.core.exceptions import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    TranscriptionError,
    AnalysisError,
)
from app.core.validators import FileValidator

logger = logging.getLogger(__name__)


class SpeechAnalysisPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        analyzer: SpeechAnalyzer,
        gigachat_client: Optional[GigaChatClient] = None,
        enable_cache: bool = True,
        enable_metrics: bool = True,
        include_timings: bool = True,  # Новая опция
    ):
        self.audio_extractor = AdvancedFfmpegAudioExtractor()
        self.transcriber = transcriber
        self.analyzer = analyzer
        self.gigachat_client = gigachat_client
        self.include_timings = include_timings

        # Ограничение параллельных анализов
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(
            settings.max_concurrent_analyses)

        # Инициализация кеша
        self.cache = None
        if enable_cache:
            cache_dir = Path(settings.cache_dir) / "analysis"
            self.cache = AnalysisCache(cache_dir, ttl_seconds=settings.cache_ttl)

        # Инициализация сборщика метрик
        self.metrics_collector = None
        if enable_metrics:
            self.metrics_collector = MetricsCollector(
                Path("logs/metrics.jsonl"))

    @cache_analysis(ttl_hours=1)
    async def analyze_upload(self, file: UploadFile) -> EnhancedAnalysisResult:
        """
        Анализирует загруженное видео с поддержкой таймингов.
        """
        # Ограничиваем параллельность: берём семафор
        await self._semaphore.acquire()
        try:
            # Начинаем сбор метрик
            if self.metrics_collector:
                await self._start_metrics_collection(file)

            # Валидация файла
            await self._validate_file(file)

            # Создаем временные файлы
            temp_video_path, temp_audio_path = await self._create_temp_files(file)

            try:
                # 1) Извлечение аудио
                if self.metrics_collector:
                    self.metrics_collector.start_subtask("audio_extraction")

                await self._extract_audio(temp_video_path, temp_audio_path)

                if self.metrics_collector:
                    self.metrics_collector.end_subtask("audio_extraction")

                # 2) Транскрибация (с таймингами слов)
                if self.metrics_collector:
                    self.metrics_collector.start_subtask("transcription")

                transcript = await self._transcribe_audio(temp_audio_path)

                if self.metrics_collector:
                    self.metrics_collector.end_subtask("transcription")

                # 3) Анализ с таймингами
                if self.metrics_collector:
                    self.metrics_collector.start_subtask("analysis")

                result = await self._analyze_speech(transcript, temp_audio_path)

                if self.metrics_collector:
                    self.metrics_collector.end_subtask("analysis")
                    # Обновляем длительность в метриках
                    if hasattr(result, 'duration_sec'):
                        self.metrics_collector._metrics["duration_sec"] = result.duration_sec

                # 4) Расширенный анализ через GigaChat
                if self.gigachat_client and settings.gigachat_enabled:
                    result = await self._enhance_with_gigachat(result)

                # Завершаем сбор метрик успехом
                if self.metrics_collector:
                    self.metrics_collector.end_processing(success=True)

                logger.info(f"Анализ завершен успешно: {file.filename}")
                return result
            finally:
                # Очистка временных файлов
                self._cleanup_temp_files(temp_video_path, temp_audio_path)
        except Exception as e:
            # Завершаем сбор метрик с ошибкой
            if self.metrics_collector:
                self.metrics_collector.end_processing(
                    success=False, error_message=str(e))
            # Пробрасываем исключение дальше
            raise
        finally:
            # Релизуем семафор в любом случае
            try:
                self._semaphore.release()
            except Exception:
                pass

    async def _start_metrics_collection(self, file: UploadFile):
        """Начинает сбор метрик"""
        if not self.metrics_collector:
            return

        # Читаем размер файла
        file_size = 0
        try:
            # Сохраняем текущую позицию
            current_pos = await file.tell()

            # Переходим в конец
            await file.seek(0, 2)
            file_size = await file.tell()

            # Возвращаемся на место
            await file.seek(current_pos)
        except Exception as e:
            logger.warning(f"Не удалось определить размер файла: {e}")

        self.metrics_collector.start_processing(
            filename=file.filename or "unknown",
            file_size=file_size
        )

    async def _validate_file(self, file: UploadFile) -> None:
        """Валидирует загруженный файл"""
        if not file.filename:
            raise UnsupportedFileTypeError(
                file_extension="unknown",
                allowed_extensions=settings.allowed_video_extensions
            )

        # Проверка расширения
        file_ext = Path(file.filename).suffix.lower()
        if not file_ext or file_ext not in settings.allowed_video_extensions:
            raise UnsupportedFileTypeError(
                file_extension=file_ext,
                allowed_extensions=settings.allowed_video_extensions
            )

        # Проверка размера
        await self._validate_file_size(file)

    async def _validate_file_size(self, file: UploadFile) -> None:
        """Проверяет размер файла"""
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024

        try:
            # Пытаемся получить размер из атрибута
            if hasattr(file, 'size') and file.size is not None:
                file_size = file.size
            else:
                # Читаем файл для определения размера
                content = await file.read(max_size_bytes + 1)
                file_size = len(content)

                # Возвращаем указатель в начало
                await file.seek(0)

            if file_size > max_size_bytes:
                file_size_mb = file_size / (1024 * 1024)
                raise FileTooLargeError(
                    file_size_mb=file_size_mb,
                    max_size_mb=settings.max_file_size_mb
                )

            logger.info(f"Файл валиден: {file.filename}, размер: {file_size / (1024 * 1024):.2f} MB")

        except Exception as e:
            logger.error(f"Ошибка проверки размера файла: {e}")
            # Не блокируем выполнение из-за ошибки определения размера

    async def _create_temp_files(self, file: UploadFile) -> tuple[Path, Path]:
        """Создает временные файлы для обработки"""
        suffix = Path(file.filename or "video").suffix or ".mp4"

        # Создаем временный видеофайл
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_video_path = Path(tmp_video.name)
        tmp_video.close()

        # Сохраняем загруженный файл
        await self._save_upload_to_path(file, temp_video_path)

        # Путь для аудиофайла
        temp_audio_path = temp_video_path.with_suffix(".wav")

        return temp_video_path, temp_audio_path

    async def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        """Извлекает аудио из видео"""
        logger.info(f"Извлечение аудио из {video_path.name}")

        try:
            # Extract audio in thread pool to avoid blocking event loop
            await asyncio.to_thread(self.audio_extractor.extract, video_path, audio_path, 300)

            # Дополнительная валидация аудиофайла
            if audio_path.exists():
                file_size = audio_path.stat().st_size
                if file_size == 0:
                    raise AnalysisError("Извлеченный аудиофайл пуст")
                
                # Добавляем проверку на минимальный размер аудиофайла для предотвращения анализа пустых аудио
                if file_size < 1024:  # Меньше 1KB - скорее всего пустое аудио
                    # Проверим, есть ли действительно звук в аудио
                    is_valid_audio, error_msg = await self._validate_audio_content(audio_path)
                    if not is_valid_audio:
                        raise AnalysisError(f"Аудиофайл не содержит речи или слишком короткий: {error_msg}")
                
                logger.info(f"Аудио извлечено: {file_size:,} байт")
            else:
                raise AnalysisError("Аудиофайл не был создан")

        except TimeoutException as e:
            logger.error(f"Таймаут извлечения аудио: {e}")
            raise AnalysisError(
                "Извлечение аудио заняло слишком много времени")
        except Exception as e:
            logger.error(f"Ошибка извлечения аудио: {e}")
            raise AnalysisError(f"Не удалось извлечь аудио: {str(e)}")
    
    async def _validate_audio_content(self, audio_path: Path) -> tuple[bool, str]:
        """Проверяет содержимое аудиофайла на наличие речи"""
        try:
            import wave
            import numpy as np
            
            # Открываем wav файл и проверяем его содержимое
            with wave.open(str(audio_path), 'r') as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
                duration = frames / float(sample_rate)
                
                if duration < 0.1:  # Меньше 100 мс - слишком короткое аудио
                    return False, "Аудио слишком короткое для анализа"
                
                # Читаем аудиоданные
                raw_audio = wf.readframes(frames)
                
                # Конвертируем в numpy array для анализа
                if wf.getsampwidth() == 2:  # 16-bit
                    audio_array = np.frombuffer(raw_audio, dtype=np.int16)
                elif wf.getsampwidth() == 4:  # 32-bit
                    audio_array = np.frombuffer(raw_audio, dtype=np.int32)
                else:
                    return False, "Неподдерживаемый формат аудио"
                
                # Проверяем уровень громкости (RMS)
                rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                
                # Если средняя громкость очень низкая, это может быть тишина
                if rms < 50:  # Порог для определения "тишины"
                    return False, "Аудио содержит в основном тишину"
                
                return True, "Аудио содержит звук"
        except Exception as e:
            logger.warning(f"Ошибка проверки содержимого аудио: {e}")
            return False, f"Ошибка проверки аудио: {str(e)}"

    async def _transcribe_audio(self, audio_path: Path):
        """Транскрибирует аудио (с таймингами слов)"""
        logger.info("Транскрибация аудио с таймингами слов...")

        try:
            # Валидация аудиофайла перед транскрибацией
            is_valid, error_msg = FileValidator.validate_audio_file(audio_path)
            if not is_valid:
                logger.warning(f"Аудиофайл не прошел валидацию: {error_msg}")

            # Run transcription in thread pool (faster-whisper is blocking)
            transcript = await asyncio.to_thread(self.transcriber.transcribe, audio_path)

            if not transcript.segments or not transcript.text.strip():
                logger.warning("Транскрипт пуст или содержит только пробелы")

            if transcript.word_timings:
                logger.info(f"Транскрибация завершена: {len(transcript.segments)} сегментов, {len(transcript.word_timings)} таймингов слов")
            else:
                logger.warning("Транскрибация не вернула тайминги слов")

            return transcript

        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")
            raise TranscriptionError(
                f"Не удалось транскрибировать аудио: {str(e)}")

    async def _analyze_speech(self, transcript, audio_path: Path) -> EnhancedAnalysisResult:
        """Анализирует речь с таймингами"""
        logger.info("Анализ метрик речи с таймингами...")

        try:
            # Run analyzer with async support for contextual analysis
            result = await self.analyzer.analyze(
                transcript,
                audio_path,
                self.include_timings,
                gigachat_client=self.gigachat_client,
                cache=self.cache
            )

            # Дополнительная проверка результата
            if result.words_total == 0:
                logger.warning("В результате анализа 0 слов")

            if result.timed_data.word_timings_count > 0:
                logger.info(f"Анализ с таймингами: {result.timed_data.word_timings_count} слов, {len(
                    result.timed_data.filler_words_detailed)} слов-паразитов, {len(result.timed_data.pauses_detailed)} пауз")

            logger.info(f"Анализ завершен: {result.words_total} слов, темп: {result.words_per_minute:.1f} WPM")

            # Optionally use LLM to classify filler words in context
            if self.gigachat_client and settings.llm_fillers_enabled and result.timed_data.filler_words_detailed:
                try:
                    contexts = []
                    # Build context windows around each filler
                    for filler in result.timed_data.filler_words_detailed:
                        # Find index of nearest word timing in the transcript
                        ft = filler.timestamp
                        nearest_idx = None
                        min_diff = float('inf')
                        for i, wt in enumerate(transcript.word_timings):
                            diff = abs(wt.start - ft)
                            if diff < min_diff:
                                min_diff = diff
                                nearest_idx = i
                        # context: two words before and two after
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
                        # Apply classification results
                        for idx, cl in enumerate(classified):
                            if idx < len(result.timed_data.filler_words_detailed):
                                fw = result.timed_data.filler_words_detailed[idx]
                                fw.context_score = cl.get("score")
                                fw.is_context_filler = cl.get("is_filler", False)

                except Exception as e:
                    logger.warning(f"LLM filler classification failed: {e}")
            return result

        except Exception as e:
            logger.error(f"Ошибка анализа речи: {e}")
            raise AnalysisError(f"Не удалось проанализировать речь: {str(e)}")

    async def _enhance_with_gigachat(self, result: EnhancedAnalysisResult) -> EnhancedAnalysisResult:
        """Добавляет анализ от GigaChat"""
        logger.info("Запрос расширенного анализа через GigaChat...")

        try:
            # Создаем базовый AnalysisResult для GigaChat
            base_result = AnalysisResult(
                duration_sec=result.duration_sec,
                speaking_time_sec=result.speaking_time_sec,
                speaking_ratio=result.speaking_ratio,
                words_total=result.words_total,
                words_per_minute=result.words_per_minute,
                filler_words=result.filler_words,
                pauses=result.pauses,
                phrases=result.phrases,
                advice=result.advice,
                transcript=result.transcript,
                gigachat_analysis=None
            )

            gigachat_analysis = await self.gigachat_client.analyze_speech(base_result)

            if gigachat_analysis:
                logger.info(f"GigaChat анализ получен: {gigachat_analysis.overall_assessment[:100]}...")

                # Обновляем результат с анализом GigaChat
                result.gigachat_analysis = gigachat_analysis
            else:
                logger.warning("GigaChat вернул пустой результат")

        except Exception as e:
            logger.error(f"Ошибка GigaChat анализа: {e}")
            # Продолжаем без анализа GigaChat

        return result

    @staticmethod
    async def _save_upload_to_path(upload: UploadFile, dst: Path) -> None:
        """Сохраняет загруженный файл"""
        await upload.seek(0)
        # Используем aiofiles для асинхронной записи на диск
        chunk_size = 1024 * 1024  # 1 MB
        async with aiofiles.open(str(dst), "wb") as out_file:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                await out_file.write(chunk)
        logger.info(f"Файл сохранен: {dst}")

    @staticmethod
    def _cleanup_temp_files(*paths: Path) -> None:
        """Удаляет временные файлы"""
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Удален временный файл: {path}")
            except Exception as e:
                logger.warning(
                    f"Не удалось удалить временный файл {path}: {e}")
