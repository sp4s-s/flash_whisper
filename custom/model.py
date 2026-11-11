"""Batch-1 FP16 Whisper: GPU log-mel, HF encoder, and a CUDA-graphed decoder built on custom kernels.

The decoder runs one token per step with static buffers: fused residual + LayerNorm, split-K single-query
attention over a static KV cache (the self attention cache write is fused in), and a fused LM head GEMV + argmax that keeps the token on the GPU.
Projections and the MLP stay on cuBLAS.
The encoder and the cross K/V projection are captured into their own CUDA graph. The host syncs once for TTFT and then once per CHUNK steps.
"""

import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

from custom.ext import load_ext
from custom.selftest import SPLIT_KEYS, run_selftest, workspace

SAMPLE_RATE = 16000
N_SAMPLES = 30 * SAMPLE_RATE
PROMPT = ('<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|notimestamps|>')
CHUNK = 8  # decode steps per graph replay; costs at most CHUNK - 1 wasted steps after EOS


class CustomModel:
    @torch.inference_mode()
    def __init__(self, model_id, cache_dir):
        if not torch.cuda.is_available():
            raise RuntimeError('CustomModel requires a CUDA device')
        self.ext = load_ext()
        hf = WhisperForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
        self.encoder = hf.model.encoder
        self.features = WhisperFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.window = torch.hann_window(self.features.n_fft, device='cuda')
        self.mel_filters = torch.from_numpy(self.features.mel_filters).float().cuda()

        cfg, dec, gen = hf.config, hf.model.decoder, hf.generation_config
        self.heads, self.head_dim = cfg.decoder_attention_heads, cfg.d_model // cfg.decoder_attention_heads
        self.scale = self.head_dim ** -0.5
        self.max_pos, self.enc_len = cfg.max_target_positions, cfg.max_source_positions
        self.prompt = self.tokenizer.convert_tokens_to_ids(list(PROMPT))
        self.max_new = self.max_pos - len(self.prompt)
        eos = gen.eos_token_id if gen.eos_token_id is not None else cfg.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos)

        self.embed, self.pos_embed = dec.embed_tokens.weight, dec.embed_positions.weight
        self.lm_head = hf.proj_out.weight.contiguous()
        self.final_ln = (dec.layer_norm.weight, dec.layer_norm.bias, dec.layer_norm.eps)
        self.layers = [self._pack_layer(layer) for layer in dec.layers]
        cross = [layer.encoder_attn for layer in dec.layers]
        self.cross_w = torch.cat([torch.cat([a.k_proj.weight, a.v_proj.weight]) for a in cross]).contiguous()
        self.cross_b = torch.cat([torch.cat([torch.zeros_like(a.v_proj.bias), a.v_proj.bias]) for a in cross])

        vocab, n_layers, half = self.lm_head.shape[0], len(self.layers), torch.float16
        self.base_mask = torch.zeros(vocab, dtype=torch.bool, device='cuda')
        self.base_mask[list(gen.suppress_tokens or [])] = True
        self.first_mask = self.base_mask.clone()
        self.first_mask[list(gen.begin_suppress_tokens or [])] = True
        self.suppress = self.base_mask.clone()

        self.self_kv = torch.zeros(n_layers, 2, self.heads, self.max_pos, self.head_dim, dtype=half, device='cuda')
        self.cross_kv = torch.zeros(n_layers, 2, self.heads, self.enc_len, self.head_dim, dtype=half, device='cuda')
        self.h = torch.zeros(1, cfg.d_model, dtype=half, device='cuda')
        self.y = torch.zeros_like(self.h)
        self.attn = torch.zeros(cfg.d_model, dtype=half, device='cuda')
        self.cur = torch.zeros(1, dtype=torch.long, device='cuda')
        self.state = torch.zeros(2, dtype=torch.long, device='cuda')  # [position, generated count]
        self.tokens = torch.zeros(self.max_pos, dtype=torch.long, device='cuda')
        self.best = torch.zeros(1, dtype=torch.long, device='cuda')
        self.workspace = workspace(self.heads, -(-max(self.max_pos, self.enc_len) // SPLIT_KEYS))
        self.mel = torch.zeros(1, cfg.num_mel_bins, 2 * self.enc_len, dtype=half, device='cuda')

        self.selftest = run_selftest(self.ext)
        self.selftest['mel_max_abs_diff'] = self._mel_diff()
        self.graph_one, self.graph_chunk = self._capture()
        self.graph_enc = self._capture_encoder()

    @staticmethod
    def _pack_layer(layer):
        sa, ca = layer.self_attn, layer.encoder_attn
        ln = lambda m: (m.weight, m.bias, m.eps)
        return SimpleNamespace(
            ln1=ln(layer.self_attn_layer_norm), ln2=ln(layer.encoder_attn_layer_norm), ln3=ln(layer.final_layer_norm),
            qkv_w=torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]).contiguous(),
            qkv_b=torch.cat([sa.q_proj.bias, torch.zeros_like(sa.q_proj.bias), sa.v_proj.bias]),
            out=(sa.out_proj.weight, sa.out_proj.bias), cross_q=(ca.q_proj.weight, ca.q_proj.bias),
            cross_out=(ca.out_proj.weight, ca.out_proj.bias),
            fc1=(layer.fc1.weight, layer.fc1.bias), fc2=(layer.fc2.weight, layer.fc2.bias))

    def _log_mel(self, wave):
        # Same math as WhisperFeatureExtractor._torch_extract_fbank_features, kept on the GPU.
        stft = torch.stft(wave, self.features.n_fft, self.features.hop_length, window=self.window, return_complex=True)
        mel = torch.clamp(self.mel_filters.T @ (stft[..., :-1].abs() ** 2), min=1e-10).log10()
        mel = torch.maximum(mel, mel.max() - 8.0)
        return (mel + 4.0) / 4.0

    def _mel_diff(self):
        audio = np.random.default_rng(0).standard_normal(10 * SAMPLE_RATE).astype(np.float32) * 0.1
        ref = self.features(audio, sampling_rate=SAMPLE_RATE, return_tensors='pt').input_features[0].cuda()
        wave = torch.zeros(N_SAMPLES, device='cuda')
        wave[:audio.size] = torch.from_numpy(audio).cuda()
        return float((self._log_mel(wave) - ref).abs().max())

    def _step(self):
        ext, state = self.ext, self.state
        pos = state[:1]
        torch.add(self.embed.index_select(0, self.cur), self.pos_embed.index_select(0, pos), out=self.h)
        pending = None
        for i, layer in enumerate(self.layers):
            ext.add_layernorm(self.h, pending, *layer.ln1[:2], self.y, layer.ln1[2])
            q, k, v = F.linear(self.y, layer.qkv_w, layer.qkv_b).view(3, self.heads, 1, self.head_dim)
            ext.decode_attention(q, self.self_kv[i, 0], self.self_kv[i, 1], self.attn, state, k, v, *self.workspace, self.scale)
            ext.add_layernorm(self.h, F.linear(self.attn.view(1, -1), *layer.out), *layer.ln2[:2], self.y, layer.ln2[2])
            q = F.linear(self.y, *layer.cross_q)
            ext.decode_attention(q, self.cross_kv[i, 0], self.cross_kv[i, 1], self.attn, None, None, None, *self.workspace, self.scale)
            ext.add_layernorm(self.h, F.linear(self.attn.view(1, -1), *layer.cross_out), *layer.ln3[:2], self.y, layer.ln3[2])
            pending = F.linear(F.gelu(F.linear(self.y, *layer.fc1)), *layer.fc2)
        ext.add_layernorm(self.h, pending, *self.final_ln[:2], self.y, self.final_ln[2])
        ext.lm_head_argmax(self.y, self.lm_head, self.suppress, self.best)
        ext.argmax_finalize(self.best, self.tokens, self.state, self.cur, self.max_pos)

    def _capture(self):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        graphs = []
        for steps in (1, CHUNK):
            self.state.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(steps):
                    self._step()
            graphs.append(graph)
        torch.cuda.synchronize()
        return graphs

    def _encode(self):
        enc = self.encoder(self.mel).last_hidden_state[0]
        kv = F.linear(enc, self.cross_w, self.cross_b)
        self.cross_kv.copy_(kv.view(self.enc_len, len(self.layers), 2, self.heads, self.head_dim).permute(1, 2, 3, 0, 4))
        return enc

    def _capture_encoder(self):
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._encode()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.enc_out = self._encode()
            self.mel.normal_()
            eager = self._encode().clone()
            graph.replay()
            self.selftest['encoder_graph_max_abs_diff'] = float((self.enc_out.float() - eager.float()).abs().max())
            self.selftest['encoder_graph'] = True
            return graph
        except Exception as exc:
            self.selftest['encoder_graph'] = False
            self.selftest['encoder_graph_error'] = f'{type(exc).__name__}: {exc}'
            return None
        finally:
            torch.cuda.synchronize()

    @torch.inference_mode()
    def transcribe(self, audio):
        if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1:
            raise ValueError('audio must be a one-dimensional float32 NumPy array')
        if not 0 < audio.size <= N_SAMPLES or not np.isfinite(audio).all():
            raise ValueError('audio must contain finite samples and be between 0 and 30 seconds')
        torch.cuda.synchronize()
        start = time.perf_counter()

        wave = torch.zeros(N_SAMPLES, device='cuda')
        wave[:audio.size] = torch.from_numpy(audio).cuda()
        mel = self._log_mel(wave)
        torch.cuda.synchronize()
        mel_s = time.perf_counter() - start

        self.mel.copy_(mel[None])
        if self.graph_enc:
            self.graph_enc.replay()
        else:
            self._encode()
        torch.cuda.synchronize()
        encoder_s = time.perf_counter() - start - mel_s

        decode_start = time.perf_counter()
        self.state.zero_()
        last = len(self.prompt) - 1
        for i, token in enumerate(self.prompt):
            self.cur.fill_(token)
            if i == last:
                self.suppress.copy_(self.first_mask)
            self.graph_one.replay()
            if i == last:
                self.suppress.copy_(self.base_mask)
            else:
                self.state[1].zero_()  # predictions made inside the prompt are discarded
        token_ids = [int(self.tokens[0])]
        ttft = time.perf_counter() - start
        while token_ids[-1] not in self.eos and len(token_ids) < self.max_new:
            self.graph_chunk.replay()
            for token in self.tokens[len(token_ids):len(token_ids) + CHUNK].tolist():
                token_ids.append(token)
                if token in self.eos or len(token_ids) >= self.max_new:
                    break
        ttlt = time.perf_counter() - start
        decode_s = time.perf_counter() - decode_start

        detok_start = time.perf_counter()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return {
            'text': text, 'token_ids': token_ids, 'ttft_s': ttft, 'ttlt_s': ttlt,
            'e2e_s': time.perf_counter() - start, 'mel_s': mel_s, 'encoder_s': encoder_s,
            'decode_s': decode_s, 'detok_s': time.perf_counter() - detok_start, 'n_tokens': len(token_ids),
        }
