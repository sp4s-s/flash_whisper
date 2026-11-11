"""Remote backend subprocess. Writes progress after every measured request."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path
from uuid import uuid4

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

import numpy as np

from reporting import save_json


PROMPT = '<|startoftranscript|><|en|><|transcribe|><|notimestamps|>'
SAMPLE_RATE = 16000
MODEL_REVISION = '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'


def prepare_data(out):
    """CPU-only: decode the frozen LibriSpeech set once into clips.npz + references.json."""
    from eval.clips import load_clips
    clips = load_clips(manifest=out / 'clips.json')
    np.savez(out / 'clips.npz', **{f'w{i}': clip['waveform'] for i, clip in enumerate(clips)})
    save_json(out / 'references.json', {clip['id']: clip['reference'] for clip in clips})
    print(f'prepared {len(clips)} clips')


def load_prepared(data):
    ids = json.loads((data / 'clips.json').read_text())
    references = json.loads((data / 'references.json').read_text())
    with np.load(data / 'clips.npz') as arrays:
        return [{'id': key, 'reference': references[key], 'waveform': arrays[f'w{i}']} for i, key in enumerate(ids)]


def concatenate_speech(clips, offset, seconds):
    target = seconds * SAMPLE_RATE
    pieces = []
    size = 0
    position = offset
    while size < target:
        audio = clips[position % len(clips)]['waveform']
        pieces.append(audio[:target - size])
        size += len(pieces[-1])
        position += 1
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


class VllmModel:
    def __init__(self, model_id, profile_only=False):
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM

        if profile_only:
            from vllm.config import ProfilerConfig

        self.params = SamplingParams(temperature=0, max_tokens=444)
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=model_id, dtype='half', gpu_memory_utilization=0.5,
            max_model_len=448, max_num_seqs=1, mm_processor_cache_gb=0, max_num_batched_tokens=2048,
            **({'profiler_config': ProfilerConfig(profiler='cuda')} if profile_only else {}),
        ))

    async def transcribe(self, audio):
        start = time.perf_counter()
        prompt = {
            'encoder_prompt': {'prompt': '', 'multi_modal_data': {'audio': (audio, SAMPLE_RATE)}},
            'decoder_prompt': PROMPT,
        }
        first = last = None
        count = 0
        final = None
        async for output in self.engine.generate(prompt, self.params, request_id=str(uuid4())):
            now = time.perf_counter() - start
            if output.outputs:
                ids = output.outputs[0].token_ids
                if len(ids) > count:
                    if first is None:
                        first = now
                    last = now
                    count = len(ids)
                final = output.outputs[0]
        if final is None:
            raise RuntimeError('vLLM returned no output')
        return {'text': final.text, 'token_ids': list(final.token_ids), 'ttft_s': first,
                'ttlt_s': last, 'e2e_s': time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('data', 'vllm', 'pytorch', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', type=Path, help='directory with clips.json, clips.npz, references.json')
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.backend == 'data':
        prepare_data(out)
        return
    result = {'status': 'starting', 'timings': {}, 'model': args.model, 'backend': args.backend,
              'errors': [], 'timing_boundary': 'in-memory 16 kHz float32 audio through decoded text',
              'fairness': 'All paths use the same prompt, greedy decoding and the checkpoint suppress lists; token ids are compared in the report.',
              'environment': {'python': platform.python_version()}}
    path = out / f'{args.backend}.json'

    def checkpoint():
        save_json(path, result)

    try:
        import torch
        import transformers
        import vllm
        result['environment'].update({'torch': torch.__version__, 'cuda': torch.version.cuda,
                                      'transformers': transformers.__version__, 'vllm': vllm.__version__})
        for command in (['nvidia-smi'], ['nvcc', '--version']):
            try:
                result['environment'][command[0]] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            except (OSError, subprocess.CalledProcessError) as exc:
                result['environment'][command[0]] = str(exc)
        checkpoint()
        dataset_start = time.perf_counter()
        clips = load_prepared(args.data)
        result['dataset_load_s'] = time.perf_counter() - dataset_start
        if len(clips) < 100:
            result['errors'].append(f'Only {len(clips)} eligible clips available')
        result['status'] = 'loading model'
        checkpoint()
        from huggingface_hub import snapshot_download
        download_start = time.perf_counter()
        local_model = snapshot_download(args.model, revision=MODEL_REVISION, cache_dir='/cache/huggingface')
        result['model_revision'] = MODEL_REVISION
        result['model_download_or_cache_lookup_s'] = time.perf_counter() - download_start
        async def run():
            started = time.perf_counter()
            if args.backend == 'vllm':
                model = VllmModel(local_model, args.profile_only)
                async def transcribe(audio):
                    return await model.transcribe(audio)
            elif args.backend == 'pytorch':
                from custom.pytorch_model import PytorchModel
                model = PytorchModel(local_model, '/cache/huggingface')
                async def transcribe(audio):
                    return model.transcribe(audio)
            else:
                from custom.model import CustomModel
                model = CustomModel(local_model, '/cache/huggingface')
                result['selftest'] = model.selftest
                async def transcribe(audio):
                    return model.transcribe(audio)
            result['model_load_s'] = time.perf_counter() - started
            result['environment']['gpu'] = torch.cuda.get_device_name(0)
            transcripts = {}
            result['status'] = 'warmup'
            checkpoint()
            for i in range(1 if args.profile_only else 3):
                await transcribe(concatenate_speech(clips, i, 30))
            if args.profile_only:
                audio = concatenate_speech(clips, 1, 30)
                if args.backend == 'vllm':
                    await model.engine.start_profile()
                else:
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                try:
                    for i in range(3):
                        with torch.cuda.nvtx.range(f'{args.backend}:pass{i}'):
                            await transcribe(audio)
                finally:
                    if args.backend == 'vllm':
                        await model.engine.stop_profile()
                    else:
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                result['status'] = 'profile complete'
                return
            for seconds in (5, 15, 30):
                result['status'] = f'timing {seconds}s'
                rows = result['timings'].setdefault(str(seconds), [])
                for i in range(20):
                    audio = concatenate_speech(clips, i, seconds)
                    audio_hash = hashlib.sha256(audio.tobytes()).hexdigest()
                    row = await transcribe(audio)
                    row['audio_sha256'] = audio_hash
                    rows.append(row)
                    checkpoint()
            result['status'] = 'accuracy'
            checkpoint()
            for clip in clips:
                row = await transcribe(clip['waveform'])
                transcripts[clip['id']] = row['text']
                save_json(out / 'transcripts' / f'{args.backend}.json', transcripts)
                checkpoint()
            result['status'] = 'complete'
        asyncio.run(run())
    except Exception:
        result['status'] = 'failed'
        result['errors'].append(traceback.format_exc())
        checkpoint()
        raise
    checkpoint()


if __name__ == '__main__':
    main()
