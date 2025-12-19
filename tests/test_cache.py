"""Test transcription caching."""
import asyncio
from pathlib import Path

from app.services.transcriber import LocalWhisperTranscriber
from app.core.config import settings


def test_transcription_cache():
    """Test transcription caching."""
    print("Creating transcriber with caching...")

    transcriber = LocalWhisperTranscriber(
        cache_dir=Path(settings.cache_dir),
        cache_ttl=settings.cache_ttl
    )

    print(f"Model: {transcriber.model_size}")
    print(f"Cache dir: {transcriber.cache_dir}")
    print(f"TTL: {transcriber.cache_ttl} seconds")

    test_audio = Path("test_empty.wav")

    # Create simple WAV file header
    with open(test_audio, "wb") as f:
        f.write(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    try:
        print(f"Transcribing {test_audio}...")
        result1 = transcriber.transcribe(test_audio)
        print("First transcription completed")

        print(f"Retranscribing {test_audio}...")
        result2 = transcriber.transcribe(test_audio)
        print("Second transcription completed")

        # Check that results are identical (cache was used)
        print(f"Results identical: {result1.text == result2.text}")

        # Check cache files
        cache_files = list(transcriber.cache_dir.glob("*.pkl"))
        print(f"Cache files: {len(cache_files)}")

    finally:
        # Cleanup
        if test_audio.exists():
            test_audio.unlink()
