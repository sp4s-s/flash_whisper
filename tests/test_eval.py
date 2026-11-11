import json

import numpy as np
import pytest

from eval.clips import SAMPLE_RATE, select_clips
from eval.wer import score_saved, score_wer


def row(key, seconds):
    return {"id": key, "text": key, "audio": {"array": np.ones(seconds * SAMPLE_RATE), "sampling_rate": SAMPLE_RATE}}


def test_selection_is_deterministic_and_manifest_is_authoritative(tmp_path):
    rows = [row("a", 1), row("b", 2), row("long", 31)]
    manifest = tmp_path / "clips.json"
    first = select_clips(rows, manifest)
    assert [clip["id"] for clip in first] == json.loads(manifest.read_text())
    assert set(json.loads(manifest.read_text())) == {"a", "b"}
    assert [clip["id"] for clip in select_clips(reversed(rows), manifest)] == [clip["id"] for clip in first]
    assert all(clip["waveform"].dtype == np.float32 for clip in first)
    with pytest.raises(ValueError):
        select_clips(rows[:1], manifest)


def test_wer_normalizes_and_aggregates_by_words():
    scores = score_wer({"a": "Hello, WORLD!", "b": "apple banana cherry"}, {"a": "hello world", "b": "apple banana"})
    assert scores["per_clip"]["a"] == 0
    assert scores["per_clip"]["b"] == pytest.approx(1 / 3)
    assert scores["aggregate"] == pytest.approx(1 / 5)
    with pytest.raises(ValueError):
        score_wer({"a": "hello"}, {"b": "hello"})


def test_selection_accepts_audio_decoder(tmp_path):
    class Tensor:
        def mean(self, dim):
            assert dim == 0
            return self

        def cpu(self):
            return np.ones(SAMPLE_RATE, dtype=np.float32)

    class Decoder:
        def get_all_samples(self):
            return type('Samples', (), {'data': Tensor(), 'sample_rate': SAMPLE_RATE})()

    clip = select_clips([{'id': 'a', 'text': 'a', 'audio': Decoder()}], tmp_path / 'clips.json')[0]
    assert clip['waveform'].shape == (SAMPLE_RATE,)


def test_score_saved_skips_missing_backends(tmp_path):
    (tmp_path / 'transcripts').mkdir()
    (tmp_path / 'references.json').write_text(json.dumps({'a': 'hello world'}))
    for name in ('vllm', 'cuda'):
        (tmp_path / 'transcripts' / f'{name}.json').write_text(json.dumps({'a': 'hello world'}))
    scored = score_saved(tmp_path)
    assert scored['vllm'] == 0 and scored['cuda'] == 0 and 'pytorch' not in scored
