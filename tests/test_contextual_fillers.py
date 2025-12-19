import pytest
from app.services.contextual_filler_analyzer import ContextualFillerAnalyzer
from app.models.transcript import Transcript, WordTiming
from app.services.cache import AnalysisCache


class DummyGigachat:
    def __init__(self, responses):
        # responses: dict mapping exact_word -> (is_filler, confidence, reason, suggestion)
        self.responses = responses

    async def classify_fillers_context(self, contexts, cache=None):
        out = []
        for i, c in enumerate(contexts):
            key = c.get('exact_word')
            resp = self.responses.get(key, (False, 0.0, 'default', None))
            enriched = dict(**c)
            enriched.update({
                'index': i + 1,
                'is_filler': resp[0],
                'confidence': resp[1],
                'reason': resp[2],
                'suggestion': resp[3]
            })
            out.append(enriched)
        return out


@pytest.mark.asyncio
async def test_contextual_filler_non_filler():
    # 'да' used as confirmation -> should be non-filler
    wt = [WordTiming(word='Да', start=0.0, end=0.2), WordTiming(word='это', start=0.3, end=0.6)]
    transcript = Transcript(text='Да это', segments=[], word_timings=wt)

    dummy = DummyGigachat({'Да': (False, 0.9, 'confirmation', None)})
    analyzer = ContextualFillerAnalyzer(dummy, cache=None)
    res = await analyzer.analyze_fillers_with_context(transcript)

    # 'Да' should be present but marked as not context filler
    assert any(r.exact_word.lower() == 'да' for r in res)
    for r in res:
        if r.exact_word.lower() == 'да':
            assert getattr(r, 'is_context_filler', False) is False
            assert abs(getattr(r, 'context_score', 0.0) - 0.9) < 1e-6


@pytest.mark.asyncio
async def test_contextual_filler_true_filler():
    # 'ну' used as filler in an utterance -> should be filler
    wt = [
        WordTiming(word='Я', start=0.0, end=0.2),
        WordTiming(word='ну', start=0.3, end=0.6),
        WordTiming(word='говорю', start=0.7, end=1.1)
    ]
    transcript = Transcript(text='Я ну говорю', segments=[], word_timings=wt)

    dummy = DummyGigachat({'ну': (True, 0.95, 'filler', 'replace_with_pause')})
    analyzer = ContextualFillerAnalyzer(dummy, cache=None)
    res = await analyzer.analyze_fillers_with_context(transcript)

    found = [r for r in res if r.exact_word.lower() == 'ну']
    assert found, 'Expected "ну" in results'
    r = found[0]
    assert getattr(r, 'is_context_filler', False) is True
    assert abs(getattr(r, 'context_score', 0.0) - 0.95) < 1e-6
