"""Result formatting shared by the remote worker and the local entrypoint."""

import csv
import datetime
import io
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


GPU_CPUS = 8
GPU_MEMORY_GB = 32


def cost_usd(seconds):
    """Modal list price: H100 + GPU_CPUS physical cores + memory, per second."""
    return seconds * (0.001097 + GPU_CPUS * 0.0000131 + GPU_MEMORY_GB * 0.00000222)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def stats(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return {'median': statistics.median(values), 'p90': percentile(values, .9)} if values else None


BACKENDS = ('vllm', 'pytorch', 'cuda')
LABELS = {'vllm': 'vLLM', 'pytorch': 'PyTorch', 'cuda': 'CUDA'}


def check_pairs(timings, backend):
    """Compare the first 3 timed reps per duration against vLLM."""
    pairs = []
    for seconds in ('5', '15', '30'):
        pairs.extend(zip(timings.get('vllm', {}).get(seconds, [])[:3], timings.get(backend, {}).get(seconds, [])[:3]))
    full = len(pairs) == 9
    return {'checked_pairs': len(pairs),
            'input_hashes_match': all(a.get('audio_sha256') == b.get('audio_sha256') for a, b in pairs) if full else None,
            'token_ids_match': all(a.get('token_ids') == b.get('token_ids') for a, b in pairs) if full else None}


def summarize(results):
    timings = results.get('timings', {})
    def rows(backend, duration):
        return timings.get(backend, {}).get(str(duration), [])
    def med(backend, duration, key):
        item = stats(rows(backend, duration), key)
        return item['median'] if item else None
    def ratio(numerator, denominator):
        return f'{numerator / denominator:.2f}x' if numerator is not None and denominator is not None and denominator > 0 else 'unavailable'
    lines = [
        '# Whisper H100 benchmark', '',
        'Latency starts with an in-memory waveform and ends with the transcript.', '',
        '| Seconds | Backend | E2E median / p90 (ms) | TTFT median / p90 (ms) | TTLT median / p90 (ms) | Audio s/s | Reps |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for duration in (5, 15, 30):
        for backend in BACKENDS:
            data = rows(backend, duration)
            def fmt(key):
                item = stats(data, key)
                return f"{item['median'] * 1000:.1f} / {item['p90'] * 1000:.1f}" if item else 'unavailable'
            e2e = med(backend, duration, 'e2e_s')
            throughput = f'{duration / e2e:.2f}' if e2e and e2e > 0 else 'unavailable'
            lines.append(f'| {duration} | {LABELS[backend]} | {fmt("e2e_s")} | {fmt("ttft_s")} | {fmt("ttlt_s")} | {throughput} | {len(data)} |')
    lines += ['', '## Speedup (median; >1 means faster)', '',
              '| Seconds | PyTorch E2E vs vLLM | PyTorch TTFT vs vLLM | CUDA E2E vs vLLM | CUDA TTFT vs vLLM | CUDA E2E vs PyTorch |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for duration in (5, 15, 30):
        def vs(base, other, key='e2e_s'):
            return ratio(med(base, duration, key), med(other, duration, key))
        lines.append(f'| {duration} | {vs("vllm", "pytorch")} | {vs("vllm", "pytorch", "ttft_s")} | '
                     f'{vs("vllm", "cuda")} | {vs("vllm", "cuda", "ttft_s")} | {vs("pytorch", "cuda")} |')
    lines += ['', '## Cost per audio hour (30 s clips, GPU list price)', '', '| Backend | Audio s/s | USD per audio hour |', '| --- | ---: | ---: |']
    for backend in BACKENDS:
        e2e = med(backend, 30, 'e2e_s')
        speed = 30 / e2e if e2e and e2e > 0 else None
        lines.append(f'| {LABELS[backend]} | ' + (f'{speed:.2f} | {cost_usd(1) * 3600 / speed:.2f}' if speed else 'unavailable | unavailable') + ' |')
    lines += ['', '## Acceptance (against vLLM)', '']
    checks = results.get('correctness', {})
    wer = results.get('wer', {})
    vllm = rows('vllm', 30)
    for backend in ('pytorch', 'cuda'):
        label = LABELS[backend]
        mine = rows(backend, 30)
        check = checks.get(backend, {})
        comparable = check.get('token_ids_match') is True and check.get('input_hashes_match') is True
        if backend == 'cuda':
            comparable = comparable and checks.get('custom_cuda_verified') is True
        complete_reps = len(vllm) >= 20 and len(mine) >= 20
        v, c = med('vllm', 30, 'e2e_s'), med(backend, 30, 'e2e_s')
        if v and c and c > 0:
            verdict = ('pass' if v / c > 1.5 else 'fail') if comparable and complete_reps else 'unverified'
            lines.append(f'- {label} 30-second speedup: {v / c:.2f}x; {verdict} (target > 1.5x).')
        else:
            lines.append(f'- {label} 30-second speedup: unavailable.')
        vt, ct = med('vllm', 30, 'ttft_s'), med(backend, 30, 'ttft_s')
        if vt and ct:
            verdict = ('pass' if ct < vt else 'fail') if comparable and complete_reps else 'unverified'
            lines.append(f'- {label} TTFT: {ct * 1000:.1f} ms vs {vt * 1000:.1f} ms; {verdict}.')
        else:
            lines.append(f'- {label} TTFT: unavailable.')
        if all(isinstance(wer.get(k), (int, float)) for k in ('vllm', backend)):
            diff = 100 * (wer[backend] - wer['vllm'])
            verdict = ('pass' if diff <= 1 else 'fail') if comparable and wer.get('num_clips', 0) >= 100 else 'unverified'
            lines.append(f'- {label} WER difference: {diff:.3f} percentage points; {verdict}.')
        else:
            lines.append(f'- {label} WER: unavailable.')
    lines += ['', '## Status', '', f"- {results.get('status', 'incomplete')}"]
    for backend in ('pytorch', 'cuda'):
        check = checks.get(backend, {})
        lines.append(f'- {LABELS[backend]} vs vLLM: input hashes match: {check.get("input_hashes_match", "unavailable")}; token IDs match: {check.get("token_ids_match", "unavailable")}.')
    for error in results.get('errors', []):
        lines.append(f'- {error}')
    lines += ['', 'Timings are not comparable unless decoding and token correctness checks pass.', '']
    return '\n'.join(lines)


def build_report(root: Path) -> str:
    root = Path(root)
    lines = []
    def read(path):
        return json.loads(path.read_text())
    try:
        result = read(root / 'results.json')
    except (OSError, ValueError):
        result = {}
    backends = {}
    for name in BACKENDS:
        try:
            backends[name] = read(root / f'{name}.json')
        except (OSError, ValueError):
            backends[name] = {}
    def section(name, fn):
        lines.append(f'\n{name}')
        try:
            fn()
        except Exception as exc:
            lines.append(f'{name}: unavailable ({exc})')
    def median(values):
        values = [v for v in values if isinstance(v, (int, float))]
        return statistics.median(values) if values else 'unavailable'
    def rows(name, duration='30'):
        return backends[name].get('timings', {}).get(duration, [])
    def header():
        env = next((b.get('environment', {}) for b in backends.values() if b.get('environment', {}).get('gpu')), {})
        lines.extend([f"run_id: {result.get('run_id', root.name)}", f'UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}',
                      f"code_sha256: {result.get('code_sha256', 'unavailable')}", f"status: {result.get('status', 'unavailable')}",
                      f"remote_wall_seconds: {result.get('remote_wall_seconds', 'unavailable')}",
                      f"estimated cost USD: {cost_usd(result.get('remote_wall_seconds', 0)):.4f}" if isinstance(result.get('remote_wall_seconds'), (int, float)) else 'estimated cost USD: unavailable', f"GPU: {env.get('gpu', 'unavailable')}"])
    def environment():
        for name, data in backends.items():
            env = data.get('environment', {})
            lines.append(f'{name}: ' + ', '.join(f'{k}={env.get(k, "unavailable")}' for k in ('python', 'torch', 'cuda', 'vllm', 'transformers')))
            lines.extend(str(env.get('nvidia-smi', 'unavailable')).splitlines()[:15])
    def errors():
        for name, data in [('results', result), *backends.items()]:
            for error in data.get('errors', []):
                lines.append(f'{name}: ' + '\n'.join(str(error).splitlines()[-40:]))
    def stages():
        decode = {}
        for name in BACKENDS:
            data = rows(name)
            if name == 'vllm':
                keys = ('e2e_s', 'ttft_s', 'ttlt_s')
            else:
                keys = ('mel_s', 'encoder_s', 'decode_s', 'detok_s', 'n_tokens')
            lines.append(f'{name}: ' + ', '.join(f'{k}={median([r.get(k) for r in data])}' for k in keys))
            if name != 'vllm':
                e2e = median([r.get('e2e_s') for r in data])
                parts = [median([r.get(k) for r in data]) for k in keys[:4]]
                if isinstance(e2e, (int, float)) and e2e > 0 and all(isinstance(v, (int, float)) for v in parts):
                    shares = [100 * v / e2e for v in (*parts, e2e - sum(parts))]
                    lines.append(f'{name} stage share of median e2e: ' + ', '.join(f'{k}={v:.1f}%' for k, v in zip(('mel', 'encoder', 'decode', 'detok', 'other'), shares)))
                else:
                    lines.append(f'{name} stage share of median e2e: unavailable')
            vals = []
            for r in data:
                n = len(r.get('token_ids', []))
                if name != 'vllm' and n and isinstance(r.get('decode_s'), (int, float)):
                    vals.append(1000 * r['decode_s'] / n)
                if name == 'vllm' and n > 1 and all(isinstance(r.get(k), (int, float)) for k in ('ttft_s', 'ttlt_s')):
                    vals.append(1000 * (r['ttlt_s'] - r['ttft_s']) / (n - 1))
            decode[name] = median(vals)
            lines.append(f'{name} decode ms/token median: {decode[name]}')
        for name in ('pytorch', 'cuda'):
            v, c = decode['vllm'], decode[name]
            value = f'{v / c:.2f}x' if isinstance(v, (int, float)) and isinstance(c, (int, float)) and c > 0 else 'unavailable'
            lines.append(f'decode ms/token ratio vllm/{name}: {value}')
    def correctness():
        for name in ('pytorch', 'cuda'):
            for duration in ('5', '15', '30'):
                pairs = list(zip(rows('vllm', duration)[:3], rows(name, duration)[:3]))
                lines.append(f'{name} {duration}s pairs={len(pairs)} input hash match={all(a.get("audio_sha256") == b.get("audio_sha256") for a,b in pairs) if pairs else "unavailable"} token-id match={all(a.get("token_ids") == b.get("token_ids") for a,b in pairs) if pairs else "unavailable"}')
                if duration == '30':
                    for i, (a, b) in enumerate(pairs):
                        if a.get('token_ids') != b.get('token_ids'):
                            lines.extend([f'rep {i + 1} vllm: {str(a.get("text", ""))[:200]}', f'rep {i + 1} {name}: {str(b.get("text", ""))[:200]}'])
    def custom_kernels():
        selftest = backends['cuda'].get('selftest')
        lines.append('selftest: ' + json.dumps(selftest, indent=2))
        summary = root / 'ptxas' / 'summary.txt'
        lines.append('ptxas:')
        lines.extend(summary.read_text(errors='replace').splitlines()[:40] if summary.exists() else ['unavailable'])
        metrics = ('Duration', 'DRAM Throughput', 'Memory Throughput', 'Compute (SM) Throughput',
                   'Registers Per Thread', 'Achieved Occupancy', 'Theoretical Occupancy', 'Block Size', 'Grid Size')
        for name in ('lm_head_argmax_kernel', 'argmax_finalize_kernel', 'decode_attention_split_kernel', 'decode_attention_combine_kernel', 'add_layernorm_kernel'):
            rep = root / 'ncu' / f'{name}.ncu-rep'
            failure = root / 'ncu' / f'{name}-FAILED.txt'
            lines.append(f'{name}:')
            if not rep.exists():
                lines.append(failure.read_text(errors='replace').strip() if failure.exists() else 'profile unavailable')
                continue
            try:
                proc = subprocess.run(['ncu', '--import', str(rep), '--page', 'details', '--csv'],
                                      capture_output=True, text=True, timeout=120, check=True)
                matches = [row for row in csv.reader(io.StringIO(proc.stdout)) if any(metric in cell for metric in metrics for cell in row)]
                lines.extend(', '.join(row) for row in matches)
                if not matches:
                    lines.append('requested metrics unavailable')
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                lines.append(f'import unavailable: {exc}')
    def wer():
        data = result.get('wer', {})
        lines.append(f'clips: {data.get("num_clips", "unavailable")}')
        for name in BACKENDS:
            lines.append(f'{name} aggregate: {data.get(name, "unavailable")}')
            for key, value in sorted(data.get(f'{name}_per_clip', {}).items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f'{name} {key}: {value}')
        for name in ('pytorch', 'cuda'):
            if all(isinstance(data.get(n), (int, float)) for n in ('vllm', name)):
                lines.append(f'difference {name} - vllm: {100 * (data[name] - data["vllm"]):.3f} percentage points')
    def clocks():
        with (root / 'clocks.csv').open() as f:
            data = list(csv.reader(f))
        for i, label in ((1, 'SM clock MHz'), (2, 'power W'), (3, 'temperature C')):
            values = []
            for row in data[1:]:
                try:
                    values.append(float(re.search(r'-?\d+(?:\.\d+)?', row[i]).group()))
                except (IndexError, AttributeError, ValueError):
                    continue
            lines.append(f'{label} min/median/max: {min(values)} / {median(values)} / {max(values)}' if values else f'{label}: unavailable')
    def nsys():
        for name in BACKENDS:
            rep = root / 'nsys' / f'{name}.nsys-rep'
            if not rep.exists():
                failure = root / 'nsys' / f'{name}-FAILED.txt'
                if not failure.exists():
                    failure = root / 'nsys' / 'FAILED.txt'
                lines.append(f'{name}: {failure.read_text().strip() if failure.exists() else "profile unavailable"}')
                continue
            for report, limit in (('cuda_gpu_kern_sum', 25), ('cuda_api_sum', 10)):
                try:
                    proc = subprocess.run(['nsys', 'stats', '--report', report, '--format', 'csv', str(rep)], capture_output=True, text=True, timeout=120, check=True)
                    parsed = list(csv.DictReader(io.StringIO(proc.stdout[proc.stdout.find('Time ('):])))
                    parsed.sort(key=lambda r: float((r.get('Total Time (ns)') or '0').replace(',', '')), reverse=True)
                    lines.append(f'{name} {report}:')
                    for row in parsed[:limit]:
                        lines.append(f"{(row.get('Name') or '')[:90]} total_ms={float((row.get('Total Time (ns)') or '0').replace(',', '')) / 1e6:.3f} instances={row.get('Instances')} avg_us={float((row.get('Avg (ns)') or '0').replace(',', '')) / 1000:.3f}")
                    if report == 'cuda_gpu_kern_sum':
                        total_ms = sum(float((r.get('Total Time (ns)') or '0').replace(',', '')) for r in parsed) / 1e6
                        lines.append(f'{name} GPU kernel ms per pass: {total_ms / 3:.3f}')
                except Exception as exc:
                    lines.append(f'{name} {report}: unavailable ({exc})')
            try:
                proc = subprocess.run(['nsys', 'stats', '--report', 'nvtx_sum', '--format', 'csv', str(rep)],
                                      capture_output=True, text=True, timeout=120, check=True)
                parsed = list(csv.DictReader(io.StringIO(proc.stdout[proc.stdout.find('Time ('):])))
                passes = {}
                for row in parsed:
                    label = row.get('Range') or row.get('Name') or ''
                    if re.search(rf'{name}:pass[0-2]$', label):
                        passes[label.split(':')[-1]] = float((row.get('Total Time (ns)') or '0').replace(',', '')) / 1e6
                if len(passes) == 3:
                    lines.append(f'{name} NVTX range ms per pass: ' + ', '.join(f'{k}={v:.3f}' for k, v in sorted(passes.items())))
                else:
                    lines.append(f'{name} NVTX range ms per pass: unavailable')
            except Exception as exc:
                lines.append(f'{name} NVTX range ms per pass: unavailable ({exc})')
    def logs():
        paths = [Path('/cache/data/prepare.log'), *(root / f'{name}.log' for name in BACKENDS), *sorted((root / 'nsys').glob('*.log'))]
        for path in paths:
            if not path.exists():
                continue
            clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', path.read_text(errors='replace')).splitlines()
            matches = [line[:300] for line in clean if re.search(r'error|warn|traceback|fatal|backend|attention|cuda graph|compil|loading weights|model loading took|kv cache|memory', line, re.I)]
            lines.append(f'{path.name} matches:')
            lines.extend(matches[:60])
            lines.append(f'{path.name} tail:')
            lines.extend(line[:300] for line in clean[-40:])
    section('HEADER', header)
    section('ENVIRONMENT', environment)
    section('ERRORS', errors)
    section('BENCHMARK TABLE', lambda: lines.extend(summarize(result).splitlines()))
    section('PER-STAGE BREAKDOWN', stages)
    section('CORRECTNESS', correctness)
    section('CUSTOM KERNELS', custom_kernels)
    section('WER', wer)
    section('GPU CLOCKS', clocks)
    section('NSYS', nsys)
    section('LOG TAILS', logs)
    section('BUDGET', lambda: lines.extend([next((line for line in lines if line.startswith('estimated cost USD:')), 'estimated cost USD: unavailable'), 'Check the Modal dashboard.']))
    return '\n'.join(lines) + '\n'
