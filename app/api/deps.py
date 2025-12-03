from functools import lru_cache

from app.services.audio_extractor import FfmpegAudioExtractor
from app.services.transcriber import LocalWhisperTranscriber
from app.services.analyzer import SpeechAnalyzer
from app.services.pipeline import SpeechAnalysisPipeline


@lru_cache(maxsize=1)
def get_speech_pipeline() -> SpeechAnalysisPipeline:
    """
    Создаёт и кеширует единственный экземпляр пайплайна.
    """
    extractor = FfmpegAudioExtractor()
    transcriber = LocalWhisperTranscriber()
    analyzer = SpeechAnalyzer()
    return SpeechAnalysisPipeline(
        audio_extractor=extractor,
        transcriber=transcriber,
        analyzer=analyzer,
    )
