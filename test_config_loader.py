import json
import os
import tempfile
import unittest
from pathlib import Path

from config import ConfigError, DEFAULT_CONFIG, load_config, validate_config


class ConfigValidationTests(unittest.TestCase):
    def test_validate_config_accepts_valid_config(self):
        config = {
            "api_url": "http://localhost:8000/v1/chat/completions",
            "model": "test-model",
            "max_iterations": 10,
            "temperature": 0.7,
        }
        result = validate_config(config)
        self.assertEqual(result["api_url"], config["api_url"])
        self.assertEqual(result["model"], config["model"])
        self.assertEqual(result["max_iterations"], 10)
        self.assertEqual(result["temperature"], 0.7)

    def test_validate_config_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ConfigError, "(?i)unknown"):
            validate_config({"max_iteratons": 10})

    def test_validate_config_rejects_non_dict(self):
        with self.assertRaises(ConfigError):
            validate_config("not a dict")
        with self.assertRaises(ConfigError):
            validate_config([])

    def test_validate_config_rejects_empty_api_url(self):
        with self.assertRaises(ConfigError):
            validate_config({"api_url": ""})
        with self.assertRaises(ConfigError):
            validate_config({"api_url": "   "})

    def test_validate_config_rejects_invalid_api_url_scheme(self):
        with self.assertRaises(ConfigError):
            validate_config({"api_url": "ftp://localhost:8000"})

    def test_validate_config_rejects_invalid_model(self):
        with self.assertRaises(ConfigError):
            validate_config({"model": ""})
        with self.assertRaises(ConfigError):
            validate_config({"model": 123})

    def test_validate_config_rejects_invalid_max_iterations(self):
        with self.assertRaises(ConfigError):
            validate_config({"max_iterations": 0})
        with self.assertRaises(ConfigError):
            validate_config({"max_iterations": -1})
        with self.assertRaises(ConfigError):
            validate_config({"max_iterations": 1.5})
        with self.assertRaises(ConfigError):
            validate_config({"max_iterations": True})

    def test_validate_config_rejects_invalid_temperature(self):
        with self.assertRaises(ConfigError):
            validate_config({"temperature": -0.1})
        with self.assertRaises(ConfigError):
            validate_config({"temperature": 2.1})
        with self.assertRaises(ConfigError):
            validate_config({"temperature": "0.5"})
        with self.assertRaises(ConfigError):
            validate_config({"temperature": True})

    def test_validate_config_rejects_invalid_max_tokens(self):
        with self.assertRaises(ConfigError):
            validate_config({"max_tokens": 0})
        with self.assertRaises(ConfigError):
            validate_config({"max_tokens": -5})


class ConfigLoaderTests(unittest.TestCase):
    def test_load_config_returns_defaults(self):
        config = load_config()
        self.assertEqual(config["api_url"], DEFAULT_CONFIG["api_url"])
        self.assertEqual(config["model"], DEFAULT_CONFIG["model"])
        self.assertEqual(config["max_iterations"], DEFAULT_CONFIG["max_iterations"])

    def test_load_config_from_json_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"model": "custom-model", "max_iterations": 15}, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            self.assertEqual(config["model"], "custom-model")
            self.assertEqual(config["max_iterations"], 15)
            # Defaults should still be present for other values
            self.assertEqual(config["api_url"], DEFAULT_CONFIG["api_url"])
        finally:
            os.unlink(temp_path)

    def test_load_config_rejects_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/path/config.json")

    def test_load_config_rejects_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with self.assertRaises(ConfigError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_config_rejects_non_object_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([1, 2, 3], f)
            temp_path = f.name

        try:
            with self.assertRaises(ConfigError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_config_from_env_vars(self):
        old_env = os.environ.get("MLX_MODEL")
        old_max = os.environ.get("MLX_MAX_ITERATIONS")
        try:
            os.environ["MLX_MODEL"] = "env-model"
            os.environ["MLX_MAX_ITERATIONS"] = "25"
            config = load_config()
            self.assertEqual(config["model"], "env-model")
            self.assertEqual(config["max_iterations"], 25)
        finally:
            if old_env is not None:
                os.environ["MLX_MODEL"] = old_env
            else:
                os.environ.pop("MLX_MODEL", None)
            if old_max is not None:
                os.environ["MLX_MAX_ITERATIONS"] = old_max
            else:
                os.environ.pop("MLX_MAX_ITERATIONS", None)

    def test_load_config_rejects_invalid_env_var(self):
        old_env = os.environ.get("MLX_MAX_ITERATIONS")
        try:
            os.environ["MLX_MAX_ITERATIONS"] = "-5"
            with self.assertRaises(ConfigError):
                load_config()
        finally:
            if old_env is not None:
                os.environ["MLX_MAX_ITERATIONS"] = old_env
            else:
                os.environ.pop("MLX_MAX_ITERATIONS", None)

    def test_load_config_applies_overrides_last(self):
        config = load_config(model="override-model", max_tokens=1024)
        self.assertEqual(config["model"], "override-model")
        self.assertEqual(config["max_tokens"], 1024)

    def test_load_config_rejects_invalid_overrides(self):
        with self.assertRaises(ConfigError):
            load_config(temperature=5.0)

    def test_load_config_overrides_take_precedence_over_env(self):
        old_env = os.environ.get("MLX_MODEL")
        try:
            os.environ["MLX_MODEL"] = "env-model"
            config = load_config(model="override-model")
            self.assertEqual(config["model"], "override-model")
        finally:
            if old_env is not None:
                os.environ["MLX_MODEL"] = old_env
            else:
                os.environ.pop("MLX_MODEL", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
