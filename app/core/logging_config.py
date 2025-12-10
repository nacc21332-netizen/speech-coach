"""
Конфигурация логирования для приложения.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import coloredlogs


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
):
    """
    Настраивает логирование для приложения.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу для записи логов (опционально)
        max_file_size: Максимальный размер файла лога
        backup_count: Количество резервных копий
    """
    # Устанавливаем уровень
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Форматер
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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
        file_handler.setFormatter(file_formatter)
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
        root_logger.info(f"Logs will be written to: {
                         log_file} (max {max_file_size//1024//1024}MB)")
