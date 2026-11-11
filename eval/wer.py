"""Whisper-normalized word error rates."""

import jiwer
import json
from pathlib import Path
from whisper.normalizers import EnglishTextNormalizer


def score_wer(references, hypotheses):
    """Score matching ID-to-text mappings; return aggregate and per-ID WER.

    WER values are fractions, not percentages. Aggregate WER is corpus-level,
    weighted by reference word count. Both mappings must be nonempty and have
    exactly the same IDs.
    """
    if not references or references.keys() != hypotheses.keys():
        raise ValueError("references and hypotheses must have the same nonempty ID set")
    normalize = EnglishTextNormalizer()
    ids = sorted(references)
    refs = [normalize(references[key]) for key in ids]
    hyps = [normalize(hypotheses[key]) for key in ids]
    if not any(refs):
        raise ValueError("normalized references contain no words")
    return {
        "aggregate": jiwer.wer(refs, hyps),
        "per_clip": {key: jiwer.wer(ref, hyp) for key, ref, hyp in zip(ids, refs, hyps)},
    }


def score_saved(root):
    root = Path(root)
    references = json.loads((root / 'references.json').read_text())
    result = {}
    for name in ('vllm', 'pytorch', 'cuda'):
        path = root / 'transcripts' / f'{name}.json'
        if not path.exists():
            continue
        hypotheses = json.loads(path.read_text())
        scored = score_wer(references, hypotheses)
        result[name] = scored['aggregate']
        result[f'{name}_per_clip'] = scored['per_clip']
    result['num_clips'] = len(references)
    return result
