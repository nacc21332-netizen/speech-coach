"""
Конфигурация логирования для приложения.
"""
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
import coloredlogs


class JSONFormatter(logging.Formatter):
    """
    Пользовательский форматер для JSON логов.
    Преобразует логи в JSON для машинной обработки.
    """
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Добавляем дополнительные поля если они есть
        if hasattr(record, 'request_id') and record.request_id:
            log_obj["request_id"] = record.request_id
        
        # Добавляем исключение если есть
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Сериализуем в JSON
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    json_logs: bool = True
):
    """
    Настраивает логирование для приложения.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу для записи логов (опционально)
        max_file_size: Максимальный размер файла лога
        backup_count: Количество резервных копий
        json_logs: Использовать JSON форматирование для логов
    """
    # Устанавливаем уровень
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Форматер для консоли (с цветами)
    console_formatter = coloredlogs.ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        field_styles={
            'asctime': {'color': 'green'},
            'name': {'color': 'blue'},
            'levelname': {'color': 'magenta', 'bold': True},
            'message': {'color': 'white'}
        }
    )

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Очищаем существующие обработчики
    root_logger.handlers.clear()

    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # Файловый обработчик (если указан файл)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if json_logs:
            # JSON форматер для машинной обработки и инструментов мониторинга
            file_handler.setFormatter(JSONFormatter())
        else:
            # Текстовый форматер
            text_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(text_formatter)
        
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # Настраиваем логирование для библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Логируем успешную настройку
    root_logger.info(f"Logging configured with level: {log_level}")
    if log_file:
        root_logger.info(f"Logs will be written to: {log_file} (max {max_file_size//1024//1024}MB)")
