import time
import logging
from typing import Dict, Any
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Метрики обработки файла"""
    filename: str
    file_size_mb: float
    duration_sec: float
    processing_time_sec: float
    audio_extraction_time_sec: float
    transcription_time_sec: float
    analysis_time_sec: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метрики в словарь"""
        return {
            "filename": self.filename,
            "file_size_mb": self.file_size_mb,
            "duration_sec": self.duration_sec,
            "processing_time_sec": self.processing_time_sec,
            "audio_extraction_time_sec": self.audio_extraction_time_sec,
            "transcription_time_sec": self.transcription_time_sec,
            "analysis_time_sec": self.analysis_time_sec,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }


class MetricsCollector:
    """Сборщик метрик производительности"""

    def __init__(self, metrics_file: Path = None):
        self.metrics_file = metrics_file or Path("metrics.jsonl")
        self._start_time = None
        self._metrics = {}

    def start_processing(self, filename: str, file_size: int):
        """Начинает отсчет времени обработки"""
        self._start_time = time.time()
        self._metrics = {
            "filename": filename,
            "file_size_mb": file_size / (1024 * 1024),
            "start_time": self._start_time,
            "subtasks": {}
        }

    def start_subtask(self, name: str):
        """Начинает отсчет времени для подзадачи"""
        if self._start_time is None:
            return

        self._metrics["subtasks"][name] = {
            "start": time.time(),
            "end": None,
            "duration": None
        }

    def end_subtask(self, name: str):
        """Завершает отсчет времени для подзадачи"""
        if self._start_time is None or name not in self._metrics["subtasks"]:
            return

        end_time = time.time()
        subtask = self._metrics["subtasks"][name]
        subtask["end"] = end_time
        subtask["duration"] = end_time - subtask["start"]

    def end_processing(self, success: bool = True, error_message: str = ""):
        """Завершает сбор метрик и сохраняет их"""
        if self._start_time is None:
            return

        end_time = time.time()
        processing_time = end_time - self._start_time

        # Собираем метрики системы (psutil опционален)
        memory_usage_mb = 0.0
        cpu_percent = 0.0
        try:
            import psutil as _psutil
            process = _psutil.Process()
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
        except Exception:
            # psutil отсутствует или не доступен — используем значения по умолчанию
            memory_usage_mb = 0.0
            cpu_percent = 0.0

        metrics = ProcessingMetrics(
            filename=self._metrics["filename"],
            file_size_mb=self._metrics["file_size_mb"],
            duration_sec=0,  # Будет заполнено позже
            processing_time_sec=processing_time,
            audio_extraction_time_sec=self._metrics["subtasks"].get(
                "audio_extraction", {}).get("duration", 0),
            transcription_time_sec=self._metrics["subtasks"].get(
                "transcription", {}).get("duration", 0),
            analysis_time_sec=self._metrics["subtasks"].get(
                "analysis", {}).get("duration", 0),
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            success=success,
            error_message=error_message
        )

        # Сохраняем метрики асинхронно, если есть loop, иначе синхронно
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            # Запуск записи в отдельном потоке
            loop.create_task(asyncio.to_thread(self._save_metrics, metrics))
        except RuntimeError:
            # Нет запущенного цикла - записываем синхронно
            self._save_metrics(metrics)
        logger.info(f"Метрики обработки сохранены: {processing_time:.2f} секунд")

        # Сбрасываем состояние
        self._start_time = None
        self._metrics = {}

    def _save_metrics(self, metrics: ProcessingMetrics):
        """Сохраняет метрики в файл"""
        try:
            import json
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Ошибка сохранения метрик: {e}")

    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Возвращает системные метрики"""
        try:
            import psutil

            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "process_count": len(psutil.pids()),
                "uptime": time.time() - psutil.boot_time(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка получения системных метрик: {e}")
            return {}
