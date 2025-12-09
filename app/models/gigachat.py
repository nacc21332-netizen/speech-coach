from typing import List, Optional
from pydantic import BaseModel, Field


class GigaChatAnalysis(BaseModel):
    """Модель для структурированного ответа от GigaChat"""
    overall_assessment: str = Field(description="Общая оценка выступления")
    strengths: List[str] = Field(
        default_factory=list, description="Сильные стороны")
    areas_for_improvement: List[str] = Field(
        default_factory=list, description="Зоны роста")
    detailed_recommendations: List[str] = Field(
        default_factory=list, description="Конкретные рекомендации")
    key_insights: List[str] = Field(
        default_factory=list, description="Ключевые инсайты")
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Уверенность анализа (0-1)")
