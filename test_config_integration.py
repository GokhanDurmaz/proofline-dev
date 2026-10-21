import importlib
import importlib.machinery
import importlib.util
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
