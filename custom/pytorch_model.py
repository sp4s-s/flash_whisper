"""Plain PyTorch Whisper baseline: HF transformers in FP16, CPU feature extraction, manual greedy loop.

Uses the HF KV cache and syncs the host every token. No custom kernels, no CUDA graphs.
Same prompt, greedy decoding and suppress lists as the other backends.
"""

import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class PytorchModel:
    def __init__(self, model_id, cache_dir):
        if not torch.cuda.is_available():
            raise RuntimeError("PytorchModel requires a CUDA device")
        self.processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, dtype=torch.float16
        ).to("cuda").eval()
        self.prompt = [self.model.config.decoder_start_token_id] + [
            token for _, token in self.processor.get_decoder_prompt_ids(
                language="en", task="transcribe", no_timestamps=True
            )
        ]

    @torch.inference_mode()
    def transcribe(self, audio):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1:
            raise ValueError("audio must be a one-dimensional float32 NumPy array")
        if not 0 < audio.size <= 30 * 16000 or not np.isfinite(audio).all():
            raise ValueError("audio must contain finite samples and be between 0 and 30 seconds")

        mel_start = time.perf_counter()
        features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        mel_s = time.perf_counter() - mel_start
        encoder_start = time.perf_counter()
        encoder_outputs = self.model.get_encoder()(features)
        torch.cuda.synchronize()
        encoder_s = time.perf_counter() - encoder_start
        decoder_input = torch.tensor([self.prompt], device="cuda", dtype=torch.long)
        generation = self.model.generation_config
        suppressed = generation.suppress_tokens or []
        begin_suppressed = generation.begin_suppress_tokens or []
        eos = generation.eos_token_id
        if eos is None:
            eos = self.model.config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        token_ids = []
        past = None
        ttft = ttlt = None
        # The decoder's position limit includes the four prompt tokens.
        decode_start = time.perf_counter()
        for step in range(self.model.config.max_target_positions - len(self.prompt)):
            output = self.model(
                encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input,
                past_key_values=past, use_cache=True,
            )
            logits = output.logits[0, -1].float()
            if suppressed:
                logits[suppressed] = -float("inf")
            if step == 0 and begin_suppressed:
                logits[begin_suppressed] = -float("inf")
            token = int(logits.argmax().item())
            elapsed = time.perf_counter() - start
            if ttft is None:
                ttft = elapsed
            ttlt = elapsed
            token_ids.append(token)
            if token in eos_ids:
                break
            past = output.past_key_values
            decoder_input = torch.tensor([[token]], device="cuda", dtype=torch.long)

        decode_s = time.perf_counter() - decode_start
        detok_start = time.perf_counter()
        text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        return {
            "text": text, "token_ids": token_ids, "ttft_s": ttft,
            "ttlt_s": ttlt, "e2e_s": time.perf_counter() - start,
            "mel_s": mel_s, "encoder_s": encoder_s, "decode_s": decode_s,
            "detok_s": time.perf_counter() - detok_start, "n_tokens": len(token_ids),
        }
