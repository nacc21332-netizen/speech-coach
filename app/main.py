import logging
import uuid
from contextvars import ContextVar
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.api.routes.health import router as health_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.chat import router as chat_router
from app.core.lifespan import lifespan
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.exceptions import (
    SpeechCoachException,
    FileValidationError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    TranscriptionError,
    AnalysisError,
    GigaChatError,
)

# Setup logging with proper log file handling
log_file = settings.log_file or "logs/app.log"
setup_logging(
    log_level=settings.log_level,
    log_file=log_file,
    max_file_size=settings.log_max_size_mb * 1024 * 1024,
    backup_count=settings.log_backup_count,
)

logger = logging.getLogger(__name__)

# Контекст для Request ID
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

def get_request_id() -> str:
    """Получить текущий Request ID"""
    return request_id_var.get()

app = FastAPI(
    title="Speech Coach API",
    description="Сервис для анализа качества публичной речи",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Exception handlers - specific handlers before general ones

@app.exception_handler(FileTooLargeError)
async def file_too_large_handler(request: Request, exc: FileTooLargeError):
    logger.warning(f"File too large: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_type": "FileTooLargeError",
            "max_size_mb": settings.max_file_size_mb,
        },
    )


@app.exception_handler(UnsupportedFileTypeError)
async def unsupported_file_type_handler(request: Request, exc: UnsupportedFileTypeError):
    logger.warning(f"Unsupported file type: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_type": "UnsupportedFileTypeError",
            "allowed_extensions": settings.allowed_video_extensions,
        },
    )


@app.exception_handler(TranscriptionError)
async def transcription_error_handler(request: Request, exc: TranscriptionError):
    logger.error(f"Transcription error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": "Ошибка распознавания речи. Убедитесь, что в видео есть четкая речь.",
            "error_type": "TranscriptionError",
            "internal_error": exc.detail,
        },
    )


@app.exception_handler(AnalysisError)
async def analysis_error_handler(request: Request, exc: AnalysisError):
    logger.error(f"Analysis error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": "Ошибка анализа речи. Попробуйте другой файл.",
            "error_type": "AnalysisError",
            "internal_error": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(SpeechCoachException)
async def speech_coach_exception_handler(request: Request, exc: SpeechCoachException):
    logger.warning(f"SpeechCoachException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_type": exc.__class__.__name__},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_type": exc.__class__.__name__,
        },
    )

# CORS middleware - restrict to safe origins
allow_origins = ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"]
if settings.log_level == "DEBUG":
    allow_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request ID middleware (должен быть после CORS)
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Добавляет уникальный Request ID для каждого запроса"""
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    
    response = await call_next(request)
    
    # Добавляем в headers для отладки
    response.headers["X-Request-ID"] = request_id
    
    return response

# Включаем роутеры
app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "name": "Speech Coach API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }
