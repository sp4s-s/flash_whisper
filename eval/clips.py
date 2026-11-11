"""Deterministic LibriSpeech selection."""

import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
MANIFEST = Path(__file__).with_name("clips.json")


def select_clips(rows, manifest=MANIFEST):
    """Return up to 100 test-clean clips, freezing their IDs in *manifest*.

    Rows have Hugging Face's ``id``, ``text``, and ``audio`` fields. Audio is
    a mapping containing a mono array and its 16 kHz sampling rate. Existing
    manifests are authoritative; missing or invalid clips raise ValueError.
    """
    manifest = Path(manifest)
    clips = {}
    wanted = json.loads(manifest.read_text()) if manifest.exists() else None
    if wanted is not None and (not isinstance(wanted, list) or not wanted or len(wanted) > 100
                               or any(not isinstance(key, str) for key in wanted)
                               or len(set(wanted)) != len(wanted)):
        raise ValueError('manifest contains invalid or duplicate IDs')
    wanted_set = set(wanted) if wanted is not None else None
    selected = []
    seen = set()
    for row in rows:
        key = row['id']
        if key in seen:
            raise ValueError(f'duplicate clip ID: {key}')
        seen.add(key)
        if wanted_set is not None and key not in wanted_set:
            continue
        audio = row["audio"]
        if hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            waveform = np.asarray(samples.data.mean(dim=0).cpu(), dtype=np.float32)
            sample_rate = samples.sample_rate
        else:
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = audio["sampling_rate"]
        if sample_rate != SAMPLE_RATE or waveform.ndim != 1:
            raise ValueError(f"{row['id']}: expected 16 kHz mono audio")
        if not 0 < waveform.size <= 30 * SAMPLE_RATE or not np.isfinite(waveform).all():
            continue
        clip = {'id': key, 'reference': row['text'], 'waveform': waveform}
        if wanted_set is not None:
            clips[key] = clip
            if len(clips) == len(wanted_set):
                break
            continue
        rank = int.from_bytes(hashlib.sha256(f'42:{key}'.encode()).digest(), 'big')
        heapq.heappush(selected, (-rank, key, clip))
        if len(selected) > 100:
            heapq.heappop(selected)

    if wanted is not None:
        ids = wanted
        if not set(ids) <= clips.keys():
            raise ValueError('manifest contains missing or ineligible clip IDs')
    else:
        clips = {key: clip for _, key, clip in selected}
        ids = sorted(clips, key=lambda key: (hashlib.sha256(f'42:{key}'.encode()).digest(), key))
        if not ids:
            raise ValueError("no eligible clips")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(ids, indent=2) + "\n")
    return [clips[key] for key in ids]


def load_clips(manifest=MANIFEST):
    """Load LibriSpeech test-clean from Hugging Face and select the frozen set."""
    from datasets import load_dataset

    return select_clips(load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True), manifest)

