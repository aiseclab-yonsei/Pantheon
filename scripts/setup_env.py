"""Create isolated environments, select PyTorch wheels and verify installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('cu130', 'cu128', 'cu126', 'cpu')


def detect_backend():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return 'cpu'
    match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', result.stdout)
    if result.returncode or not match:
        return 'cpu'
    supported = tuple(map(int, match.groups()))
    for version, backend in [((13, 0), 'cu130'), ((12, 8), 'cu128'), ((12, 6), 'cu126')]:
        if supported >= version:
            return backend
    return 'cpu'


def run(command):
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def fingerprint(requirements, backend):
    digest = hashlib.sha256()
    for file in requirements:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    digest.update((ROOT / 'scripts/check_install.py').read_bytes())
    digest.update(f'{backend}:{sys.version_info[:2]}'.encode())
    return digest.hexdigest()


def install(kind, env_root, backend, force=False, check_only=False):
    target = env_root / kind
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    marker = target / '.pantheon-setup.json'
    requirements = [ROOT / 'requirements.txt']
    if kind == 'gpu':
        requirements += [ROOT / 'requirements-gpu.txt', ROOT / f'requirements-torch-{backend}.txt']
    signature = fingerprint(requirements, backend)
    check_command = [python, ROOT / 'scripts/check_install.py', '--mode', kind]
    if kind == 'gpu':
        check_command += ['--backend', backend]
    if check_only:
        if not python.is_file():
            raise RuntimeError(f'Environment missing: {target}. Run bash setup.sh {kind}')
        run(check_command)
        return
    if python.is_file() and not (target / 'pyvenv.cfg').is_file():
        # Old copied Conda environments contain unrelated, conflicting packages.
        # Preserve them and build the small, isolated runtime from requirements.
        backup = env_root / f'{kind}.legacy-{time.time_ns()}'
        target.rename(backup)
        print(f'Preserved legacy environment: {backup}', flush=True)
    if not python.is_file():
        print(f'Creating isolated {kind} environment: {target}', flush=True)
        venv.EnvBuilder(with_pip=True, symlinks=os.name != 'nt').create(target)
    else:
        probe = subprocess.check_output([str(python), '-c', 'import sys; print(sys.version_info.major, sys.version_info.minor)'], text=True)
        if not (3, 11) <= tuple(map(int, probe.split())) <= (3, 13):
            raise RuntimeError(f'Unsupported Python in {target}; use a new --env-root.')
    try:
        previous = json.loads(marker.read_text()) if marker.exists() else {}
    except (OSError, ValueError):
        previous = {}
    if force or previous.get('fingerprint') != signature:
        marker.unlink(missing_ok=True)
        run([python, '-m', 'pip', 'install', '--upgrade', 'pip'])
        run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements.txt'])
        if kind == 'gpu':
            # Only torch wheels use the PyTorch index; all other packages use PyPI.
            run([python, '-m', 'pip', 'install', '-r', ROOT / f'requirements-torch-{backend}.txt'])
            run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements-gpu.txt'])
    else:
        print(f'{kind}: installed requirements unchanged; checking environment.', flush=True)
    run([python, '-m', 'pip', 'check'])
    run(check_command)
    marker.write_text(json.dumps({'fingerprint': signature, 'backend': backend,
                                 'python': str(python)}, indent=2))
    freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (env_root / f'{kind}.freeze.txt').write_text(freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='auto',
                        choices=['auto', 'cpu', 'gpu', 'inference-cpu'])
    parser.add_argument('--torch-backend', default='auto', choices=['auto', *BACKENDS])
    parser.add_argument('--env-root', type=Path,
                        default=Path(os.environ.get('PANTHEON_ENV_ROOT', ROOT / 'environment')))
    parser.add_argument('--force', action='store_true', help='Rerun package installation')
    parser.add_argument('--check', action='store_true', help='Verify an existing install without changes')
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error('Use Python 3.11, 3.12 or 3.13 (set PYTHON=/path/to/python).')
    backend = detect_backend() if args.torch_backend == 'auto' else args.torch_backend
    mode = args.mode
    if mode == 'auto':
        mode = 'gpu' if backend != 'cpu' else 'cpu'
    if mode == 'inference-cpu':
        backend = 'cpu'
    if mode == 'gpu' and backend == 'cpu':
        parser.error('No supported NVIDIA runtime found. Use cpu for CSV reproduction, '
                     'inference-cpu for image inference, or --torch-backend cu130/cu128/cu126.')
    env_root = args.env_root.expanduser().resolve()
    backend_label = backend if mode != 'cpu' else 'not needed (CSV-only)'
    print(f'Python: {sys.executable}\nMode: {mode}\nPyTorch backend: {backend_label}\nEnvironments: {env_root}', flush=True)
    install('cpu', env_root, 'cpu', args.force, args.check)
    if mode in {'gpu', 'inference-cpu'}:
        install('gpu', env_root, backend, args.force, args.check)
    print('\nSetup complete. No activation is required.', flush=True)
    if env_root != ROOT / 'environment':
        print(f'Run commands with PANTHEON_ENV_ROOT="{env_root}"', flush=True)
    print('CSV demo: bash scripts/demo.sh\nImage demo: bash scripts/demo_images.sh', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
        print(f'\nSetup failed: {error}\nFix the issue and rerun the same setup command; partial installs are retried.', file=sys.stderr)
        sys.exit(1)
