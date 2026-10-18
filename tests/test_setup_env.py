"""Bootstrap checks for hardware selection and interrupted/repeated installs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / 'scripts/setup_env.py'
SPEC = importlib.util.spec_from_file_location('setup_env', FILE)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class HardwareSelection(unittest.TestCase):
    def test_cuda_wheels_match_reported_driver_runtime(self):
        for runtime, expected in [('13.0', 'cu130'), ('12.9', 'cu128'),
                                  ('12.6', 'cu126'), ('12.4', 'cpu')]:
            with self.subTest(runtime=runtime), patch.object(setup.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], 0, f'CUDA Version: {runtime}', '')):
                self.assertEqual(setup.detect_backend(), expected)

    def test_missing_or_broken_nvidia_driver_uses_cpu(self):
        with patch.object(setup.subprocess, 'run', side_effect=FileNotFoundError):
            self.assertEqual(setup.detect_backend(), 'cpu')
        with patch.object(setup.subprocess, 'run',
                          return_value=subprocess.CompletedProcess([], 1, '', 'driver failed')):
            self.assertEqual(setup.detect_backend(), 'cpu')


class InstallationRecovery(unittest.TestCase):
    def test_failed_install_is_retried_and_completed_install_skips_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'cpu'
            (target / 'bin').mkdir(parents=True)
            (target / 'bin/python').touch()
            (target / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
            marker = target / '.pantheon-setup.json'
            marker.write_text('{')  # A killed write must not permanently block setup.
            with patch.object(setup.subprocess, 'check_output', return_value='3 12\n'), \
                 patch.object(setup, 'run', side_effect=subprocess.CalledProcessError(1, 'pip')):
                with self.assertRaises(subprocess.CalledProcessError):
                    setup.install('cpu', root, 'cpu')
            self.assertFalse(marker.exists())
            with patch.object(setup.subprocess, 'check_output', return_value='3 12\n'), \
                 patch.object(setup, 'run') as run:
                setup.install('cpu', root, 'cpu')
                self.assertTrue(any('install' in list(map(str, c.args[0])) for c in run.call_args_list))
            self.assertTrue(marker.exists())
            self.assertEqual(json.loads(marker.read_text())['backend'], 'cpu')
            with patch.object(setup.subprocess, 'check_output', return_value='3 12\n'), \
                 patch.object(setup, 'run') as run:
                setup.install('cpu', root, 'cpu')
                self.assertFalse(any('install' in list(map(str, c.args[0])) for c in run.call_args_list))

    def test_copied_conda_environment_is_preserved_before_clean_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'cpu'
            (target / 'bin').mkdir(parents=True)
            (target / 'bin/python').touch()
            (target / 'keep.txt').write_text('original environment')
            with patch.object(setup.venv, 'EnvBuilder') as builder, \
                 patch.object(setup, 'run'), \
                 patch.object(setup.subprocess, 'check_output', return_value='freeze\n'):
                builder.return_value.create.side_effect = lambda path: path.mkdir(parents=True)
                setup.install('cpu', root, 'cpu')
                builder.return_value.create.assert_called_once_with(target)
            backups = list(root.glob('cpu.legacy-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / 'keep.txt').read_text(), 'original environment')


if __name__ == '__main__':
    unittest.main()
