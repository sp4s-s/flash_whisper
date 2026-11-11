"""Builds and loads the Whisper CUDA extension. `python -m custom.ext` builds it ahead of time."""

import argparse
import os
import sys
from pathlib import Path

SOURCE = Path(__file__).parent / 'kernels' / 'whisper_kernels.cu'
BUILD_DIR = Path(os.environ.get('WHISPER_EXT_DIR', '/opt/whisper_ext'))
CUDA_FLAGS = ['-O3', '-lineinfo', '-Xptxas=-v,-warn-spills', '--expt-relaxed-constexpr']


def load_ext(verbose=False):
    from torch.utils.cpp_extension import load

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(name='whisper_ext', sources=[str(SOURCE)], extra_cuda_cflags=CUDA_FLAGS,
                extra_cflags=['-O3'], build_directory=str(BUILD_DIR), verbose=verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-log', type=Path, default=BUILD_DIR / 'build.log')
    args = parser.parse_args()
    args.build_log.parent.mkdir(parents=True, exist_ok=True)
    # ptxas -v output comes from compiler subprocesses on fd 1/2, so redirect at fd level.
    with args.build_log.open('w') as log:
        saved = os.dup(1), os.dup(2)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        error = None
        try:
            # Importing can fail on a GPU-less build host; the compiled .so is what matters.
            load_ext(verbose=True)
        except Exception as e:
            error = e
        finally:
            sys.stdout.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
    lines = args.build_log.read_text(errors='replace').splitlines()
    built = list(BUILD_DIR.glob('whisper_ext*.so'))
    if not built:
        print(f'build failed: {error!r}', file=sys.stderr)
        print('\n'.join(lines[-120:]), file=sys.stderr)
        sys.exit(1)
    for line in lines:
        if 'ptxas info' in line or 'spill' in line.lower():
            print(line)
    print(f'built: {built}')


if __name__ == '__main__':
    main()
