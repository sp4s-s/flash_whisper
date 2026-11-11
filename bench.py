"""Run the Whisper H100 benchmark with ``modal run bench.py``."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import modal

from reporting import BACKENDS, GPU_CPUS, GPU_MEMORY_GB, build_report, check_pairs, cost_usd, save_json, summarize

APP_NAME = 'whisper-h100-bench'
VOLUME_NAME = 'whisper-t4-bench-data'
MODEL = 'openai/whisper-large-v3-turbo'

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry('nvidia/cuda:13.0.3-devel-ubuntu24.04', add_python='3.12')
    .apt_install('git', 'ffmpeg', 'libsndfile1', 'cuda-nsight-systems-13-0', 'cuda-nsight-compute-13-0')
    .add_local_file('requirements.lock', '/tmp/requirements.lock', copy=True)
    .run_commands('pip install -r /tmp/requirements.lock')
    .run_commands('command -v nsys || ln -s $(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | tail -1) /usr/local/bin/nsys',
                  'command -v ncu || ln -s $(ls -d /opt/nvidia/nsight-compute/*/ncu | tail -1) /usr/local/bin/ncu')
    .env({'TORCH_CUDA_ARCH_LIST': '9.0a', 'PYTHONPATH': '/app',
          'TOKENIZERS_PARALLELISM': 'false', 'WHISPER_EXT_DIR': '/opt/whisper_ext'})
    .add_local_dir('custom', '/app/custom', copy=True)
    .add_local_dir('eval', '/app/eval', copy=True)
    .add_local_file('runner.py', '/app/runner.py', copy=True)
    .add_local_file('reporting.py', '/app/reporting.py', copy=True)
    .run_commands('cd /app && python -m custom.ext --build-log /opt/whisper_ext/build.log')
)


DATA_DIR = '/cache/data'
# Set at runtime only: the volume mounts at /cache, so image build steps must not write there.
CACHE_ENV = {'HF_HOME': '/cache/huggingface', 'VLLM_CACHE_ROOT': '/cache/vllm',
             'TORCHINDUCTOR_CACHE_DIR': '/cache/inductor', 'TRITON_CACHE_DIR': '/cache/triton',
             'XDG_CACHE_HOME': '/cache/xdg'}


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={'/cache': volume})
def prepare_data():
    """Decode the frozen clip set on CPU once; GPU runs only read clips.npz."""
    os.environ.update(CACHE_ENV)
    data = Path(DATA_DIR)
    if all((data / name).exists() for name in ('clips.json', 'clips.npz', 'references.json')):
        return 'cached'
    data.mkdir(parents=True, exist_ok=True)
    with (data / 'prepare.log').open('w') as log:
        code = subprocess.run(['python', '-X', 'faulthandler', '/app/runner.py', '--backend', 'data',
                               '--output', str(data), '--model', MODEL], stdout=log, stderr=subprocess.STDOUT).returncode
    volume.commit()
    if code:
        raise RuntimeError(f'data preparation exited with code {code}; see {DATA_DIR}/prepare.log')
    return 'prepared'


@app.function(image=image, gpu='H100', cpu=GPU_CPUS, memory=GPU_MEMORY_GB * 1024, timeout=3600, volumes={'/cache': volume})
def benchmark(run_id: str, code_sha256: str):
    os.environ.update(CACHE_ENV)
    root = Path('/cache/runs') / run_id
    root.mkdir(parents=True, exist_ok=True)
    result = {'status': 'starting', 'run_id': run_id, 'errors': [], 'timings': {}, 'wer': {}, 'model': MODEL, 'code_sha256': code_sha256}
    started = time.monotonic()
    clock = None

    def checkpoint():
        save_json(root / 'results.json', result)
        (root / 'summary.md').write_text(summarize(result))
        volume.commit()

    try:
        volume.reload()
        for name in ('clips.json', 'references.json'):
            shutil.copyfile(Path(DATA_DIR) / name, root / name)
        checkpoint()
        try:
            clocks = (root / 'clocks.csv').open('w')
            clock = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu', '--format=csv', '-lms', '100'],
                stdout=clocks, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            result['errors'].append(f'Clock logging unavailable: {exc}')
        for backend in BACKENDS:
            result['status'] = f'running {backend}'
            checkpoint()
            # A separate process frees the first model and engine before the next one loads.
            command = ['python', '-X', 'faulthandler', '/app/runner.py', '--backend', backend,
                       '--output', str(root), '--model', MODEL, '--data', DATA_DIR]
            with (root / f'{backend}.log').open('w') as log:
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    launched = time.monotonic()
                    deadline = launched + 1500
                    while process.poll() is None:
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            volume.commit()
                            stage = root / f'{backend}.json'
                            try:
                                status = json.loads(stage.read_text()).get('status')
                            except (OSError, ValueError):
                                status = 'starting'
                            if time.monotonic() - launched > 600 and status in ('starting', 'loading model'):
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                                result['errors'].append(f'{backend} startup exceeded 600 s')
                                break
                            if time.monotonic() >= deadline:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                                result['errors'].append(f'{backend} timed out; see {backend}.log')
                                break
                    if process.returncode and not any(f'{backend} timed out' in error for error in result['errors']):
                        result['errors'].append(f'{backend} exited with code {process.returncode}; see {backend}.log')
                except OSError as exc:
                    result['errors'].append(f'{backend} could not start: {exc}')
            stage = root / f'{backend}.json'
            if stage.exists():
                data = json.loads(stage.read_text())
                result['timings'][backend] = data.get('timings', {})
                result.setdefault('backends', {})[backend] = data
                if data.get('status') != 'complete':
                    result['errors'].append(f'{backend} did not complete all cases')
            checkpoint()
        missing = [name for name in BACKENDS if not (root / 'transcripts' / f'{name}.json').exists()]
        if missing:
            result['errors'].append(f'No transcripts for {", ".join(missing)}; WER skipped for them.')
        if len(missing) < len(BACKENDS):
            try:
                from eval.wer import score_saved
                result['wer'] = score_saved(root)
                save_json(root / 'wer.json', result['wer'])
            except Exception as exc:
                result['errors'].append(f'WER unavailable: {type(exc).__name__}: {exc}')
        result['correctness'] = {
            **{name: check_pairs(result['timings'], name) for name in ('pytorch', 'cuda')},
            'custom_cuda_verified': (result.get('backends', {}).get('cuda', {}).get('selftest') or {}).get('ok') is True,
            'note': 'Token ids are compared against vLLM on three timed pairs per duration; all paths use the same prompt and greedy decoding.',
        }
        ptxas = root / 'ptxas'
        ptxas.mkdir(exist_ok=True)
        build_log = Path('/opt/whisper_ext/build.log')
        if build_log.exists():
            shutil.copyfile(build_log, ptxas / 'build.log')
            lines = build_log.read_text(errors='replace').splitlines()
            (ptxas / 'summary.txt').write_text('\n'.join(line for line in lines if 'ptxas info' in line or 'spill' in line.lower()) + '\n')
        ncu_dir = root / 'ncu'
        ncu_dir.mkdir(exist_ok=True)
        if result.get('backends', {}).get('cuda', {}).get('status') == 'complete':
            ncu = shutil.which('ncu') or ('/usr/local/cuda/bin/ncu' if Path('/usr/local/cuda/bin/ncu').exists() else None)
            for name in ('lm_head_argmax_kernel', 'argmax_finalize_kernel', 'decode_attention_split_kernel', 'decode_attention_combine_kernel', 'add_layernorm_kernel'):
                log_path = ncu_dir / f'{name}.log'
                try:
                    if ncu is None:
                        raise FileNotFoundError('ncu was not present in the image')
                    command = [ncu, '--kernel-name', f'regex:{name}', '--launch-count', '1',
                               '--section', 'SpeedOfLight', '--section', 'MemoryWorkloadAnalysis',
                               '--clock-control', 'none', '--section', 'LaunchStats', '--section', 'Occupancy', '-f', '-o', str(ncu_dir / name),
                               'python', '-m', 'custom.selftest', '--out', str(ncu_dir / 'selftest')]
                    with log_path.open('w') as log:
                        subprocess.run(command, cwd='/app', stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    tail = log_path.read_text(errors='replace').splitlines()[-20:] if log_path.exists() else []
                    (ncu_dir / f'{name}-FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n' + '\n'.join(tail) + '\n')
        trace_dir = root / 'nsys'
        trace_dir.mkdir(exist_ok=True)
        nsys = shutil.which('nsys')
        if nsys is None:
            (trace_dir / 'FAILED.txt').write_text('nsys was not present in the image.\n')
        else:
            for backend in BACKENDS:
                if result.get('backends', {}).get(backend, {}).get('status') != 'complete':
                    (trace_dir / f'{backend}-FAILED.txt').write_text('Skipped: backend benchmark did not complete.\n')
                    continue
                profile_dir = trace_dir / f'{backend}-data'
                profile_dir.mkdir(exist_ok=True)
                command = [nsys, 'profile', '--trace=cuda,nvtx,osrt', '--capture-range=cudaProfilerApi',
                           '--capture-range-end=stop', '--cuda-graph-trace=node', '--force-overwrite=true',
                           '-o', str(trace_dir / backend), 'python', '/app/runner.py',
                           '--backend', backend, '--output', str(profile_dir), '--model', MODEL,
                           '--data', DATA_DIR, '--profile-only']
                try:
                    with (trace_dir / f'{backend}.log').open('w') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
                except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    (trace_dir / f'{backend}-FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n')
                checkpoint()
        result['status'] = 'incomplete' if result['errors'] else 'completed; inspect correctness gates'
    except Exception:
        result['status'] = 'failed'
        result['errors'].append(traceback.format_exc())
    finally:
        if clock is not None:
            clock.terminate()
            try:
                clock.wait(timeout=5)
            except subprocess.TimeoutExpired:
                clock.kill()
                clock.wait()
            clocks.close()
        result['remote_wall_seconds'] = time.monotonic() - started
        checkpoint()
        try:
            (root / 'report.txt').write_text(build_report(root))
        except Exception:
            (root / 'report.txt').write_text(traceback.format_exc())
        volume.commit()
    return str(root)


def next_run():
    base = Path.cwd() / 'results'
    base.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        target = base / f'run{number:02d}'
        try:
            target.mkdir()
            return target
        except FileExistsError:
            number += 1


@app.local_entrypoint()
def main():
    ledger_path = Path.cwd() / 'budget.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if not isinstance(ledger, list):
            raise ValueError('budget.json must be a list of invocation records')
    else:
        ledger = []
    estimated_prior = sum(float(entry['estimated_cost_usd']) for entry in ledger)
    if estimated_prior >= 7:
        raise RuntimeError('Estimated development spend reached $7. Check Modal billing before a final run.')
    target = next_run()
    run_id = uuid.uuid4().hex
    digest = hashlib.sha256()
    for source in [Path('bench.py'), Path('runner.py'), Path('reporting.py'), *sorted(Path('custom').rglob('*.py')), *sorted(Path('eval').rglob('*.py'))]:
        digest.update(str(source).encode())
        digest.update(source.read_bytes())
    manifest = {'run_id': run_id, 'status': 'starting', 'code_sha256': digest.hexdigest(),
                'notice': 'Budget estimates exclude image builds and storage; they are not a Modal billing limit.'}
    (target / 'local.json').write_text(json.dumps(manifest, indent=2) + '\n')
    started = time.monotonic()
    try:
        print('data:', prepare_data.remote())
        benchmark.remote(run_id, digest.hexdigest())
    finally:
        manifest['local_wall_seconds'] = time.monotonic() - started
        remote = f'/runs/{run_id}'
        try:
            for item in volume.iterdir(remote, recursive=True):
                name = item.path.lstrip('/')
                prefix = remote.lstrip('/') + '/'
                if not name.startswith(prefix) or getattr(getattr(item, 'type', None), 'name', None) != 'FILE':
                    continue
                rel = Path(name[len(prefix):])
                if not rel.parts or '..' in rel.parts:
                    continue
                destination = target / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with destination.open('wb') as output:
                        for chunk in volume.read_file('/' + name):
                            output.write(chunk)
                except Exception as exc:
                    manifest.setdefault('download_errors', []).append(f'{name}: {exc}')
        except Exception as exc:
            manifest['download_error'] = str(exc)
        manifest['status'] = 'download attempted'
        (target / 'local.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result_file = target / 'results.json'
        data = json.loads(result_file.read_text()) if result_file.exists() else {}
        gpu_seconds = data.get('remote_wall_seconds')
        actual_remote_time_known = isinstance(gpu_seconds, (int, float))
        if not actual_remote_time_known:
            gpu_seconds = manifest['local_wall_seconds']
        # On failure, count the whole local interval as GPU time rather than assume zero spend.
        estimate = cost_usd(gpu_seconds)
        ledger.append({'run_id': run_id, 'gpu_seconds': gpu_seconds, 'estimated_cost_usd': estimate,
                       'remote_time_known': actual_remote_time_known})
        tmp = ledger_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(ledger, indent=2) + '\n')
        tmp.replace(ledger_path)
        summary = target / 'summary.md'
        if summary.exists():
            print(summary.read_text())
        if (target / 'report.txt').exists():
            print(f'Report: {target / "report.txt"}')
        print(f'Results: {target}')
