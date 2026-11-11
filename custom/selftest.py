"""Checks each custom kernel against a PyTorch reference. Also the NCU target: `python -m custom.selftest --out DIR`."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN, HEADS, HEAD_DIM, VOCAB = 1280, 20, 64, 51866


SPLIT_KEYS = 64


def workspace(heads, splits):
    """fp32 partials (max, exp-sum, value sum) for the split-K attention kernels."""
    return (torch.empty(heads * splits, device='cuda'), torch.empty(heads * splits, device='cuda'),
            torch.empty(heads * splits * HEAD_DIM, device='cuda'))


def _check(name, err, tol):
    return {'name': name, 'max_err': float(err), 'ok': bool(err <= tol)}


def _attention_ref(q, k, v, length, scale):
    s = torch.einsum('hd,hld->hl', q.float(), k[:, :length].float()) * scale
    return torch.einsum('hl,hld->hd', s.softmax(-1), v[:, :length].float())


@torch.inference_mode()
def run_selftest(ext):
    torch.manual_seed(0)
    dev, half = 'cuda', torch.float16
    checks = []

    h = torch.randn(1, HIDDEN, device=dev, dtype=half)
    x = torch.randn_like(h)
    gamma, beta = torch.randn(HIDDEN, device=dev, dtype=half), torch.randn(HIDDEN, device=dev, dtype=half)
    residual = h + x
    ref = F.layer_norm(residual.float(), (HIDDEN,), gamma.float(), beta.float(), 1e-5)
    y = torch.empty_like(h)
    ext.add_layernorm(h, x, gamma, beta, y, 1e-5)
    checks.append(_check('add_layernorm', max((y.float() - ref).abs().max(), (h - residual).abs().max()), 2e-2))

    scale = HEAD_DIM ** -0.5
    ws = workspace(HEADS, 24)
    for length in (1, 63, 64, 65, 200, 448):
        max_len, pos = 448, length - 1
        q = torch.randn(HEADS * HEAD_DIM, device=dev, dtype=half)
        k = torch.randn(HEADS, max_len, HEAD_DIM, device=dev, dtype=half)
        v = torch.randn_like(k)
        k_new = torch.randn(HEADS, 1, HEAD_DIM, device=dev, dtype=half)
        v_new = torch.randn_like(k_new)
        k_ref, v_ref = k.clone(), v.clone()
        k_ref[:, pos], v_ref[:, pos] = k_new[:, 0], v_new[:, 0]
        out = torch.empty_like(q)
        state = torch.tensor([pos, 0], device=dev, dtype=torch.long)
        ext.decode_attention(q, k, v, out, state, k_new, v_new, *ws, scale)
        ref = _attention_ref(q.view(HEADS, HEAD_DIM), k_ref, v_ref, length, scale).flatten()
        checks.append(_check(f'decode_attention_self_{length}', (out.float() - ref).abs().max(), 1e-2))
        written = max((k[:, pos] - k_new[:, 0]).abs().max(), (v[:, pos] - v_new[:, 0]).abs().max())
        untouched = max((k - k_ref).abs().max(), (v - v_ref).abs().max())
        checks.append(_check(f'kv_write_{length}', max(written, untouched), 0))

    q = torch.randn(HEADS * HEAD_DIM, device=dev, dtype=half)
    k = torch.randn(HEADS, 1500, HEAD_DIM, device=dev, dtype=half)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    ext.decode_attention(q, k, v, out, None, None, None, *ws, scale)
    ref = _attention_ref(q.view(HEADS, HEAD_DIM), k, v, 1500, scale).flatten()
    checks.append(_check('decode_attention_cross', (out.float() - ref).abs().max(), 1e-2))

    hidden = torch.randn(1, HIDDEN, device=dev, dtype=half)
    weight = torch.randn(VOCAB, HIDDEN, device=dev, dtype=half) * 0.02
    suppress = torch.rand(VOCAB, device=dev) < 0.01
    logits = F.linear(hidden, weight).float()[0].masked_fill(suppress, float('-inf'))
    best, tokens = torch.zeros(1, device=dev, dtype=torch.long), torch.zeros(8, device=dev, dtype=torch.long)
    state, cur = torch.tensor([3, 0], device=dev, dtype=torch.long), torch.zeros(1, device=dev, dtype=torch.long)
    ext.lm_head_argmax(hidden, weight, suppress, best)
    ext.argmax_finalize(best, tokens, state, cur, 448)
    token = int(tokens[0])
    # A different index is acceptable only when it ties the reference max at fp16 precision.
    gap = float(logits.max() - logits[token]) if 0 <= token < VOCAB else float('inf')
    bookkeeping = int(cur) == token and state.tolist() == [4, 1] and int(best) == 0
    checks.append(_check('lm_head_argmax', gap if bookkeeping else float('inf'), 1e-3))

    return {'ok': all(c['ok'] for c in checks), 'checks': checks}


def main():
    from custom.ext import load_ext

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = run_selftest(load_ext())
    (args.out / 'selftest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
