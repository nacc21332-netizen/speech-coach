import wave
import math
import numpy as np
import pytest
from pathlib import Path

from app.services.analyzer_advanced import AdvancedSpeechAnalyzer
from app.models.transcript import Transcript, WordTiming


@pytest.mark.asyncio
async def test_volume_emphasis_detection(tmp_path: Path):
    """Generate a WAV where one word segment is significantly louder and
    assert that a 'volume' emphasis is detected for that word."""

    sr = 16000
    words = 5
    dur_per_word = 0.5  # seconds
    total_dur = words * dur_per_word
    total_samples = int(total_dur * sr)

    # Build signal: low amplitude everywhere, but one louder segment in the middle
    t = np.linspace(0, total_dur, total_samples, endpoint=False)
    signal = np.zeros_like(t)

    for i in range(words):
        start = int(i * dur_per_word * sr)
        end = int((i + 1) * dur_per_word * sr)
        freq = 220.0 + i * 30.0
        amp = 1000.0
        # make middle word (index 2) louder
        if i == 2:
            amp = 8000.0
        signal[start:end] = amp * np.sin(2 * math.pi * freq * t[start:end])

    # Convert to int16
    sig_int16 = np.int16(np.clip(signal, -32767, 32767))

    wav_path = tmp_path / "test_volume.wav"
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sig_int16.tobytes())

    # Build transcript with word timings matching the segments above
    word_timings = []
    for i in range(words):
        start = round(i * dur_per_word, 3)
        end = round((i + 1) * dur_per_word, 3)
        word_timings.append(WordTiming(word=f"w{i}", start=start, end=end, confidence=0.9))

    transcript = Transcript(text=" ".join(w.word for w in word_timings), segments=[], word_timings=word_timings)

    analyzer = AdvancedSpeechAnalyzer()
    result = await analyzer.analyze_with_timings(transcript, audio_path=wav_path)

    # There should be at least one 'volume' emphasis around the loud word (index 2)
    volume_emphases = [e for e in result.timeline.emphases if e.type == 'volume']
    assert len(volume_emphases) >= 1, f"Expected volume emphasis, got {result.timeline.emphases}"

    # Check that one of them is close to the loud word start time
    loud_start = word_timings[2].start
    assert any(abs(e.timestamp - loud_start) < 0.2 for e in volume_emphases), "No volume emphasis near loud word"
