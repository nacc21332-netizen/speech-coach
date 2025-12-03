from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from app.api.deps import get_speech_pipeline
from app.models.timed_analysis import TimedAnalysisResult
from app.services.pipeline import SpeechAnalysisPipeline

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze", response_model=TimedAnalysisResult)
async def analyze_video(
    file: UploadFile = File(...),
    pipeline: SpeechAnalysisPipeline = Depends(get_speech_pipeline),
):
    """
    Анализ речи из видео с временными метками.

    Возвращает детализированные данные с таймингами для:
    - каждого слова-паразита
    - каждой паузы с контекстом
    - темпа речи в разных временных окнах
    - активности говорения
    - границ фраз
    - сегментов транскрипта с таймингами слов
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    try:
        result = await pipeline.analyze(file)
        return result
    except Exception as e:
        # TODO: логирование
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze video: {e}")
