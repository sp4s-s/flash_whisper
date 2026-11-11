import json

from reporting import build_report, percentile, summarize


def test_percentile_interpolates():
    assert percentile([1, 2, 3], .5) == 2
    assert percentile([], .9) is None


def test_speedup_is_unverified_without_cuda_and_parity():
    vllm = [{'e2e_s': 3.0, 'ttft_s': 1.0, 'ttlt_s': 2.0}] * 20
    custom = [{'e2e_s': 1.0, 'ttft_s': .5, 'ttlt_s': .8}] * 20
    report = summarize({'timings': {'vllm': {'30': vllm}, 'cuda': {'30': custom}},
                        'correctness': {'cuda': {'input_hashes_match': True, 'token_ids_match': True},
                                        'custom_cuda_verified': False}})
    assert 'CUDA 30-second speedup: 3.00x; unverified' in report
    assert 'CUDA TTFT: 500.0 ms vs 1000.0 ms; unverified' in report


def test_speedup_table():
    vllm = [{'e2e_s': 1.0, 'ttft_s': .04, 'ttlt_s': 1.0}] * 20
    torch_rows = [{'e2e_s': .8, 'ttft_s': .04, 'ttlt_s': .8}] * 20
    custom = [{'e2e_s': .4, 'ttft_s': .02, 'ttlt_s': .4}] * 20
    report = summarize({'timings': {'vllm': {'30': vllm}, 'pytorch': {'30': torch_rows}, 'cuda': {'30': custom}}})
    assert '| 30 | 1.25x | 1.00x | 2.50x | 2.00x | 2.00x |' in report
    assert '| 5 | unavailable | unavailable | unavailable | unavailable | unavailable |' in report


def test_three_backend_rows():
    row = [{'e2e_s': 1.0, 'ttft_s': .1, 'ttlt_s': 1.0}] * 20
    report = summarize({'timings': {name: {'30': row} for name in ('vllm', 'pytorch', 'cuda')}})
    for label in ('vLLM', 'PyTorch', 'CUDA'):
        assert f'| 30 | {label} |' in report
        assert f'| {label} | 30.00 |' in report


def test_stage_shares_and_decode_ratio(tmp_path):
    custom = {'e2e_s': 1.0, 'mel_s': .1, 'encoder_s': .2, 'decode_s': .3,
              'detok_s': .1, 'token_ids': [1, 2]}
    vllm = {'ttft_s': .1, 'ttlt_s': .4, 'token_ids': [1, 2]}
    for name, row in (('cuda', custom), ('pytorch', custom), ('vllm', vllm)):
        (tmp_path / f'{name}.json').write_text(json.dumps({'timings': {'30': [row]}}))
    report = build_report(tmp_path)
    assert 'cuda stage share of median e2e: mel=10.0%, encoder=20.0%, decode=30.0%, detok=10.0%, other=30.0%' in report
    assert 'pytorch stage share of median e2e: mel=10.0%' in report
    assert 'decode ms/token ratio vllm/cuda: 2.00x' in report


def test_cuda_kernel_report(tmp_path):
    (tmp_path / 'cuda.json').write_text(json.dumps({'selftest': {'ok': True, 'checks': {'x': {'max_err': 0}}}}))
    (tmp_path / 'ptxas').mkdir()
    (tmp_path / 'ptxas' / 'summary.txt').write_text('ptxas info: test\n')
    (tmp_path / 'ncu').mkdir()
    (tmp_path / 'ncu' / 'decode_attention_split_kernel-FAILED.txt').write_text('permission denied')
    report = build_report(tmp_path)
    assert report.index('\nCORRECTNESS\n') < report.index('\nCUSTOM KERNELS\n') < report.index('\nWER\n')
    assert '"max_err": 0' in report and 'ptxas info: test' in report and 'permission denied' in report


def test_nsys_reports_per_pass(tmp_path, monkeypatch):
    import subprocess

    (tmp_path / 'nsys').mkdir()
    (tmp_path / 'nsys' / 'cuda.nsys-rep').touch()
    def fake_run(command, **kwargs):
        report = command[command.index('--report') + 1]
        if report == 'nvtx_sum':
            output = 'Time (%),Total Time (ns),Range\n10,3000000,cuda:pass0\n10,6000000,cuda:pass1\n10,9000000,cuda:pass2\n'
        else:
            output = 'Time (%),Total Time (ns),Instances,Avg (ns),Name\n100,12000000,3,4000000,kernel\n'
        return subprocess.CompletedProcess(command, 0, stdout=output)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    report = build_report(tmp_path)
    assert 'GPU kernel ms per pass: 4.000' in report
    assert 'cuda NVTX range ms per pass: pass0=3.000, pass1=6.000, pass2=9.000' in report
