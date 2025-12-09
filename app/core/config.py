from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    # Путь к ffmpeg (по умолчанию просто "ffmpeg" из PATH)
    ffmpeg_path: str = Field(default="ffmpeg", alias="FFMPEG_PATH")

    # Настройки локального Whisper (faster-whisper)
    whisper_model: str = Field(
        default="small", alias="WHISPER_MODEL"
    )  # варианты: tiny, base, small, medium, large-v3
    whisper_device: str = Field(
        default="cpu", alias="WHISPER_DEVICE"
    )  # cpu или cuda
    whisper_compute_type: str = Field(
        default="int8", alias="WHISPER_COMPUTE_TYPE"
    )  # int8, int8_float16, float16, float32

    # Настройки GigaChat API
    gigachat_enabled: bool = Field(
        default=False, alias="GIGACHAT_ENABLED"
    )
    gigachat_api_key: Optional[SecretStr] = Field(
        default=None, alias="GIGACHAT_API_KEY"
    )
    gigachat_base_url: str = Field(
        default="https://gigachat.devices.sberbank.ru/api/v1",
        alias="GIGACHAT_BASE_URL"
    )
    gigachat_model: str = Field(
        default="GigaChat", alias="GIGACHAT_MODEL"
    )
    gigachat_timeout: int = Field(
        default=30, alias="GIGACHAT_TIMEOUT"
    )
    gigachat_max_tokens: int = Field(
        default=2000, alias="GIGACHAT_MAX_TOKENS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
