from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, field_validator
import json


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # FFmpeg configuration
    ffmpeg_path: str = Field(default="ffmpeg", alias="FFMPEG_PATH")

    # Настройки локального Whisper (faster-whisper)
    whisper_model: str = Field(
        default="small", alias="WHISPER_MODEL"
    )
    whisper_device: str = Field(
        default="cpu", alias="WHISPER_DEVICE"
    )
    whisper_compute_type: str = Field(
        default="int8", alias="WHISPER_COMPUTE_TYPE"
    )

    # Настройки GigaChat API (согласно документации)
    gigachat_enabled: bool = Field(
        default=False, alias="GIGACHAT_ENABLED"
    )
    gigachat_api_key: Optional[SecretStr] = Field(
        default=None, alias="GIGACHAT_API_KEY"
    )
    gigachat_auth_url: str = Field(
        default="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        alias="GIGACHAT_AUTH_URL"
    )
    gigachat_api_url: str = Field(
        default="https://gigachat.devices.sberbank.ru/api/v1",
        alias="GIGACHAT_API_URL"
    )
    gigachat_model: str = Field(
        default="gigachat:latest", alias="GIGACHAT_MODEL"
    )
    gigachat_timeout: int = Field(
        default=30, alias="GIGACHAT_TIMEOUT"
    )
    gigachat_max_tokens: int = Field(
        default=131072, alias="GIGACHAT_MAX_TOKENS"
    )
    gigachat_scope: str = Field(
        default="GIGACHAT_API_PERS",
        alias="GIGACHAT_SCOPE"
    )

    # Настройки валидации файлов
    max_file_size_mb: int = Field(
        default=100, alias="MAX_FILE_SIZE_MB"
    )
    allowed_video_extensions: List[str] = Field(
        default=[".mp4", ".mov", ".avi", ".mkv",
                 ".webm", ".flv", ".wmv", ".m4v"],
        alias="ALLOWED_VIDEO_EXTENSIONS"
    )

    # Настройки кеширования
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")  # 1 час
    cache_dir: str = Field(default="cache/analysis", alias="CACHE_DIR")

    # Настройки логирования
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")
    log_max_size_mb: int = Field(default=10, alias="LOG_MAX_SIZE_MB")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")

    # Настройки производительности
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")
    max_concurrent_analyses: int = Field(
        default=5, alias="MAX_CONCURRENT_ANALYSES")
    cleanup_temp_files: bool = Field(default=True, alias="CLEANUP_TEMP_FILES")
    temp_file_retention_minutes: int = Field(
        default=30, alias="TEMP_FILE_RETENTION_MINUTES")

    # Настройки детекции пауз и слов-паразитов
    min_pause_gap_sec: float = Field(default=0.5, alias="MIN_PAUSE_GAP_SEC")
    long_pause_sec: float = Field(default=2.5, alias="LONG_PAUSE_SEC")
    silence_factor: float = Field(default=0.35, alias="SILENCE_FACTOR")
    pause_segment_time_tolerance: float = Field(default=0.25, alias="PAUSE_SEGMENT_TIME_TOLERANCE")

    # VAD настройки
    use_webrtc_vad: bool = Field(default=True, alias="USE_WEBRTC_VAD")
    webrtc_vad_mode: int = Field(default=3, alias="WEBRTC_VAD_MODE")
    use_pyannote_vad: bool = Field(default=False, alias="USE_PYANNOTE_VAD")
    pyannote_model: str = Field(default="pyannote/voice-activity-detection", alias="PYANNOTE_MODEL")

    # Filler detection settings
    filler_cluster_gap_sec: float = Field(default=2.0, alias="FILLER_CLUSTER_GAP_SEC")

    # Emphasis detection settings
    emphasis_window_size: int = Field(default=7, alias="EMPHASIS_WINDOW_SIZE")
    emphasis_mad_multiplier: float = Field(default=2.5, alias="EMPHASIS_MAD_MULTIPLIER")
    emphasis_min_duration: float = Field(default=0.05, alias="EMPHASIS_MIN_DURATION")
    emphasis_pause_threshold: float = Field(default=0.8, alias="EMPHASIS_PAUSE_THRESHOLD")
    emphasis_content_boost: float = Field(default=0.2, alias="EMPHASIS_CONTENT_BOOST")

    # LLM settings for contextual filler detection
    llm_fillers_enabled: bool = Field(default=True, alias="LLM_FILLERS_ENABLED")
    llm_fillers_max_tokens: int = Field(default=256, alias="LLM_FILLERS_MAX_TOKENS")
    llm_fillers_model: str = Field(default="GigaChat", alias="LLM_FILLERS_MODEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @field_validator("max_file_size_mb")
    def validate_max_file_size(cls, v):
        if v <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be positive")
        if v > 1024:  # 1GB max
            raise ValueError("MAX_FILE_SIZE_MB cannot exceed 1024 (1GB)")
        return v

    @field_validator("allowed_video_extensions", mode="before")
    def parse_allowed_extensions(cls, v):
        """Парсит значение в список расширений"""
        if v is None:
            return cls.__fields__["allowed_video_extensions"].default

        if isinstance(v, str):
            try:
                # Сначала пробуем как JSON
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    v = parsed
                else:
                    # Если не JSON, то как строку с разделителями
                    v = [ext.strip() for ext in v.split(",") if ext.strip()]
            except json.JSONDecodeError:
                # Если не JSON, то как строку с разделителями
                v = [ext.strip() for ext in v.split(",") if ext.strip()]

        # Убедимся, что расширения начинаются с точки и в нижнем регистре
        if isinstance(v, list):
            validated = []
            for ext in v:
                if isinstance(ext, str):
                    if not ext.startswith("."):
                        ext = f".{ext}"
                    validated.append(ext.lower())
            return validated

        return v

    @field_validator("log_max_size_mb")
    def validate_log_max_size(cls, v):
        if v <= 0:
            raise ValueError("LOG_MAX_SIZE_MB must be positive")
        if v > 100:  # 100 MB max
            raise ValueError("LOG_MAX_SIZE_MB cannot exceed 100")
        return v

    @field_validator("log_backup_count")
    def validate_log_backup_count(cls, v):
        if v < 0:
            raise ValueError("LOG_BACKUP_COUNT cannot be negative")
        if v > 20:
            raise ValueError("LOG_BACKUP_COUNT cannot exceed 20")
        return v

    @field_validator("max_concurrent_analyses")
    def validate_max_concurrent_analyses(cls, v):
        if v <= 0:
            raise ValueError("MAX_CONCURRENT_ANALYSES must be positive")
        if v > 10:
            raise ValueError("MAX_CONCURRENT_ANALYSES cannot exceed 10")
        return v


settings = Settings()
