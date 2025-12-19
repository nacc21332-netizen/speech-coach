from fastapi import APIRouter, Depends
from app.core.config import settings
from app.api.deps import get_cache_manager

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Speech Coach API",
        "version": "1.0.0",
        "features": {
            "whisper": True,
            "gigachat": settings.gigachat_enabled
        }
    }


@router.get("/stats/cache")
async def cache_stats(cache_manager = Depends(get_cache_manager)):
    """Получить статистику двухуровневого кэша"""
    return {
        "cache_stats": cache_manager.stats(),
        "cache_enabled": settings.cache_enabled,
        "cache_ttl_seconds": settings.cache_ttl,
        "cache_dir": settings.cache_dir,
    }
