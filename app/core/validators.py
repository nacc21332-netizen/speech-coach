import mimetypes
from pathlib import Path
from typing import List, Tuple
import logging

try:
    import magic
    _MAGIC_AVAILABLE = True
except ImportError:
    _MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileValidator:
    """Валидатор файлов для проверки типа и содержимого"""

    @staticmethod
    def validate_video_file(
        file_path: Path,
        allowed_extensions: List[str],
        max_size_bytes: int
    ) -> Tuple[bool, str]:
        """
        Проверяет видеофайл на соответствие требованиям.

        Args:
            file_path: Путь к файлу
            allowed_extensions: Разрешенные расширения
            max_size_bytes: Максимальный размер в байтах

        Returns:
            Tuple[bool, str]: (валиден ли файл, сообщение об ошибке)
        """
        try:
            # Проверка существования файла
            if not file_path.exists():
                return False, "Файл не существует"

            # Проверка размера
            file_size = file_path.stat().st_size
            if file_size == 0:
                return False, "Файл пуст"
            if file_size > max_size_bytes:
                size_mb = file_size / (1024 * 1024)
                max_mb = max_size_bytes / (1024 * 1024)
                return False, f"Размер файла ({size_mb:.1f} MB) превышает максимальный ({max_mb} MB)"

            # Проверка расширения
            if not any(str(file_path).lower().endswith(ext.lower()) for ext in allowed_extensions):
                return False, f"Неподдерживаемое расширение. Разрешены: {', '.join(allowed_extensions)}"

            # Определение MIME-типа
            if _MAGIC_AVAILABLE:
                try:
                    mime_type = magic.from_file(str(file_path), mime=True)
                    if not mime_type.startswith('video/'):
                        logger.warning(
                            f"Файл имеет MIME-тип {mime_type}, но ожидался видео")
                except Exception as e:
                    logger.warning(f"Ошибка определения MIME-типа: {e}")
            else:
                logger.debug(
                    "Библиотека python-magic не установлена, проверка MIME-типа пропущена")

            return True, "Файл валиден"

        except Exception as e:
            logger.error(f"Ошибка валидации файла: {e}")
            return False, f"Ошибка проверки файла: {str(e)}"

    @staticmethod
    def validate_audio_file(file_path: Path) -> Tuple[bool, str]:
        """Проверяет аудиофайл"""
        if not file_path.exists():
            return False, "Аудиофайл не существует"

        try:
            import wave
            with wave.open(str(file_path), 'rb') as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()

                if channels != 1:
                    return False, f"Аудио должно быть моно, получено {channels} каналов"
                if sampwidth != 2:
                    return False, f"Неподдерживаемый формат сэмпла: {sampwidth} байт"
                if framerate != 16000:
                    return False, f"Частота дискретизации должна быть 16kHz, получено {framerate}Hz"

            return True, "Аудиофайл валиден"

        except Exception as e:
            return False, f"Ошибка проверки аудиофайла: {str(e)}"

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Очищает имя файла от потенциально опасных символов"""
        import re
        # Удаляем небезопасные символы
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        # Ограничиваем длину
        safe_name = safe_name[:255]
        return safe_name
