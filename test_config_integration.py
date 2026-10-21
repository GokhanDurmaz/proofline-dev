import importlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Clear cached modules
        for key in list(sys.modules.keys()):
            if key.startswith('local_agent') or key.startswith('config'):
                del sys.modules[key]

    def test_local_agent_loads_config_values(self):
        import local_agent
        # Should use defaults from config
        self.assertEqual(local_agent.API_URL, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(local_agent.MODEL, "mlx-community/Mellum2-12B-A2.5B-Instruct-4bit")
        self.assertEqual(local_agent.MAX_ITERATIONS, 20)
        self.assertEqual(local_agent.FINALIZATION_ITERATIONS, 4)
        self.assertEqual(local_agent.MAX_SOURCE_FILES, 8)
        self.assertEqual(local_agent.MAX_FILE_CHARS, 30000)
        self.assertEqual(local_agent.MAX_OUTPUT_CHARS, 12000)

    def test_local_agent_uses_env_config(self):
        old_env = {}
        for key in ['MLX_API_URL', 'MLX_MODEL', 'MLX_MAX_ITERATIONS', 'MLX_MAX_SOURCE_FILES']:
            old_env[key] = os.environ.get(key)
            os.environ.pop(key, None)

        try:
            os.environ['MLX_API_URL'] = 'http://test:9000/v1/chat/completions'
            os.environ['MLX_MODEL'] = 'test-model'
            os.environ['MLX_MAX_ITERATIONS'] = '30'
            os.environ['MLX_MAX_SOURCE_FILES'] = '10'

            # Clear cached modules
            for key in list(sys.modules.keys()):
                if key.startswith('local_agent') or key.startswith('config'):
                    del sys.modules[key]

            import local_agent
            self.assertEqual(local_agent.API_URL, 'http://test:9000/v1/chat/completions')
            self.assertEqual(local_agent.MODEL, 'test-model')
            self.assertEqual(local_agent.MAX_ITERATIONS, 30)
            self.assertEqual(local_agent.MAX_SOURCE_FILES, 10)
        finally:
            for key, val in old_env.items():
                if val is not None:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)
            # Clear cached modules
            for key in list(sys.modules.keys()):
                if key.startswith('local_agent') or key.startswith('config'):
                    del sys.modules[key]

    def test_local_agent_fails_on_invalid_config(self):
        old_env = os.environ.get('MLX_MAX_ITERATIONS')
        try:
            os.environ['MLX_MAX_ITERATIONS'] = '-5'
            # Clear cached modules
            for key in list(sys.modules.keys()):
                if key.startswith('local_agent') or key.startswith('config'):
                    del sys.modules[key]

            proc = subprocess.run(
                [sys.executable, '-c', 'import local_agent'],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(b'Configuration error', proc.stderr)
        finally:
            if old_env is not None:
                os.environ['MLX_MAX_ITERATIONS'] = old_env
            else:
                os.environ.pop('MLX_MAX_ITERATIONS', None)
            # Clear cached modules
            for key in list(sys.modules.keys()):
                if key.startswith('local_agent') or key.startswith('config'):
                    del sys.modules[key]


class CliConfigTests(unittest.TestCase):
    """End-to-end checks for the `--config` command-line option."""

    UNREACHABLE_URL = "http://127.0.0.1:1/v1/chat/completions"
    OTHER_UNREACHABLE_URL = "http://127.0.0.1:2/v1/chat/completions"

    def _env(self, **overrides):
        """Build a subprocess environment without inherited MLX_* settings."""
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("MLX_"):
                del env[key]
        env.update(overrides)
        return env

    def _run_agent(self, args, env=None):
        return subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "local_agent.py"), *args],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env() if env is None else env,
            timeout=120,
        )

    def _write_config(self, directory, payload):
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_cli_loads_json_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(tmp, {
                "api_url": self.UNREACHABLE_URL,
                "model": "config-file-model",
                "max_iterations": 1,
            })

            proc = self._run_agent([
                "--config", config_path,
                "--root", tmp,
                "describe this project",
            ])

            stdout = proc.stdout.decode()
            stderr = proc.stderr.decode()
            # The file values reached the running agent.
            self.assertIn(f"API: {self.UNREACHABLE_URL}", stdout)
            self.assertIn("Agent iteration 1/1", stdout)
            # The run reached the model call and failed on the unreachable URL.
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("API error", stderr)

    def test_cli_env_overrides_json_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(tmp, {
                "api_url": self.UNREACHABLE_URL,
                "max_iterations": 1,
            })

            proc = self._run_agent(
                ["--config", config_path, "--root", tmp, "describe this project"],
                env=self._env(MLX_API_URL=self.OTHER_UNREACHABLE_URL),
            )

            stdout = proc.stdout.decode()
            self.assertIn(f"API: {self.OTHER_UNREACHABLE_URL}", stdout)
            self.assertNotIn(f"API: {self.UNREACHABLE_URL}", stdout)

    def test_cli_without_config_keeps_existing_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run_agent(
                ["--root", tmp, "describe this project"],
                env=self._env(MLX_API_URL=self.UNREACHABLE_URL),
            )

            stdout = proc.stdout.decode()
            # Default max_iterations is still used when no file is given.
            self.assertIn(f"API: {self.UNREACHABLE_URL}", stdout)
            self.assertIn("Agent iteration 1/20", stdout)

    def test_cli_existing_options_remain_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(tmp, {
                "api_url": self.UNREACHABLE_URL,
                "max_iterations": 1,
            })

            proc = self._run_agent([
                "--config", config_path,
                "--root", tmp,
                "--mode", "read-only",
                "--allow-command", "echo hi",
                "describe this project",
            ])

            stderr = proc.stderr.decode()
            self.assertNotIn("unrecognized arguments", stderr)
            self.assertNotIn("error: unrecognized", stderr)
            self.assertIn(f"API: {self.UNREACHABLE_URL}", proc.stdout.decode())

            help_proc = self._run_agent(["--help"])
            self.assertEqual(help_proc.returncode, 0)
            help_text = help_proc.stdout.decode()
            for option in ("--root", "--config", "--mode", "--allow-command"):
                self.assertIn(option, help_text)

            missing_root = self._run_agent([
                "--root", str(Path(tmp) / "does-not-exist"),
                "describe this project",
            ])
            self.assertEqual(missing_root.returncode, 2)
            self.assertIn(
                "Project directory does not exist",
                missing_root.stderr.decode(),
            )

    def test_cli_rejects_missing_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "absent.json")

            proc = self._run_agent([
                "--config", missing,
                "--root", tmp,
                "describe this project",
            ])

            stderr = proc.stderr.decode()
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Configuration error", stderr)
            self.assertIn("not found", stderr)

    def test_cli_rejects_empty_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run_agent([
                "--config", "",
                "--root", tmp,
                "describe this project",
            ])

            stdout = proc.stdout.decode()
            stderr = proc.stderr.decode()
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Configuration error", stderr)
            self.assertIn("must not be empty", stderr)
            # The agent must not silently continue with the default configuration.
            self.assertNotIn("Agent iteration", stdout)

    def test_cli_rejects_directory_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run_agent([
                "--config", tmp,
                "--root", tmp,
                "describe this project",
            ])

            stdout = proc.stdout.decode()
            stderr = proc.stderr.decode()
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Configuration error", stderr)
            self.assertIn("is a directory", stderr)
            self.assertNotIn("Agent iteration", stdout)

    def test_cli_rejects_invalid_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = str(Path(tmp) / "broken.json")
            Path(config_path).write_text("{ invalid json }", encoding="utf-8")

            proc = self._run_agent([
                "--config", config_path,
                "--root", tmp,
                "describe this project",
            ])

            stderr = proc.stderr.decode()
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Configuration error", stderr)
            self.assertIn("Invalid JSON", stderr)

    def test_cli_rejects_unknown_config_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(tmp, {"max_iteratons": 5})

            proc = self._run_agent([
                "--config", config_path,
                "--root", tmp,
                "describe this project",
            ])

            stderr = proc.stderr.decode()
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Configuration error", stderr)
            self.assertIn("Unknown configuration key", stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
