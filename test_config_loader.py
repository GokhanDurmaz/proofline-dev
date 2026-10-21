import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from config import (
    ConfigError,
    DEFAULT_CONFIG,
    ENV_MAPPING,
    load_config,
    validate_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent
README_PATH = PROJECT_ROOT / "README.md"


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

    def test_validate_config_rejects_non_finite_temperature(self):
        with self.assertRaises(ConfigError):
            validate_config({"temperature": float("nan")})
        with self.assertRaises(ConfigError):
            validate_config({"temperature": float("inf")})
        with self.assertRaises(ConfigError):
            validate_config({"temperature": float("-inf")})


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

    def test_load_config_from_env_finalization_iterations(self):
        old_env = os.environ.get("MLX_FINALIZATION_ITERATIONS")
        try:
            os.environ["MLX_FINALIZATION_ITERATIONS"] = "6"
            config = load_config()
            self.assertEqual(config["finalization_iterations"], 6)
        finally:
            if old_env is not None:
                os.environ["MLX_FINALIZATION_ITERATIONS"] = old_env
            else:
                os.environ.pop("MLX_FINALIZATION_ITERATIONS", None)

    def test_load_config_rejects_invalid_env_finalization_iterations(self):
        old_env = os.environ.get("MLX_FINALIZATION_ITERATIONS")
        try:
            os.environ["MLX_FINALIZATION_ITERATIONS"] = "-1"
            with self.assertRaises(ConfigError):
                load_config()
        finally:
            if old_env is not None:
                os.environ["MLX_FINALIZATION_ITERATIONS"] = old_env
            else:
                os.environ.pop("MLX_FINALIZATION_ITERATIONS", None)

    def test_load_config_from_env_max_file_chars(self):
        old_env = os.environ.get("MLX_MAX_FILE_CHARS")
        try:
            os.environ["MLX_MAX_FILE_CHARS"] = "40000"
            config = load_config()
            self.assertEqual(config["max_file_chars"], 40000)
        finally:
            if old_env is not None:
                os.environ["MLX_MAX_FILE_CHARS"] = old_env
            else:
                os.environ.pop("MLX_MAX_FILE_CHARS", None)

    def test_load_config_rejects_invalid_env_max_file_chars(self):
        old_env = os.environ.get("MLX_MAX_FILE_CHARS")
        try:
            os.environ["MLX_MAX_FILE_CHARS"] = "0"
            with self.assertRaises(ConfigError):
                load_config()
        finally:
            if old_env is not None:
                os.environ["MLX_MAX_FILE_CHARS"] = old_env
            else:
                os.environ.pop("MLX_MAX_FILE_CHARS", None)

    def test_load_config_from_env_max_output_chars(self):
        old_env = os.environ.get("MLX_MAX_OUTPUT_CHARS")
        try:
            os.environ["MLX_MAX_OUTPUT_CHARS"] = "16000"
            config = load_config()
            self.assertEqual(config["max_output_chars"], 16000)
        finally:
            if old_env is not None:
                os.environ["MLX_MAX_OUTPUT_CHARS"] = old_env
            else:
                os.environ.pop("MLX_MAX_OUTPUT_CHARS", None)

    def test_load_config_rejects_invalid_env_max_output_chars(self):
        old_env = os.environ.get("MLX_MAX_OUTPUT_CHARS")
        try:
            os.environ["MLX_MAX_OUTPUT_CHARS"] = "abc"
            with self.assertRaises(ConfigError):
                load_config()
        finally:
            if old_env is not None:
                os.environ["MLX_MAX_OUTPUT_CHARS"] = old_env
            else:
                os.environ.pop("MLX_MAX_OUTPUT_CHARS", None)

    def test_load_config_rejects_invalid_env_temperature(self):
        old_env = os.environ.get("MLX_TEMPERATURE")
        try:
            os.environ["MLX_TEMPERATURE"] = "nan"
            with self.assertRaises(ConfigError):
                load_config()
        finally:
            if old_env is not None:
                os.environ["MLX_TEMPERATURE"] = old_env
            else:
                os.environ.pop("MLX_TEMPERATURE", None)

    def test_load_config_env_overrides_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"max_file_chars": 111}, f)
            temp_path = f.name

        old_env = os.environ.get("MLX_MAX_FILE_CHARS")
        try:
            os.environ["MLX_MAX_FILE_CHARS"] = "222"
            config = load_config(temp_path)
            self.assertEqual(config["max_file_chars"], 222)
        finally:
            if old_env is not None:
                os.environ["MLX_MAX_FILE_CHARS"] = old_env
            else:
                os.environ.pop("MLX_MAX_FILE_CHARS", None)
            os.unlink(temp_path)

    def test_load_config_rejects_empty_config_path(self):
        with self.assertRaisesRegex(ConfigError, "must not be empty"):
            load_config(config_path="")
        with self.assertRaisesRegex(ConfigError, "must not be empty"):
            load_config(config_path="   ")

    def test_load_config_rejects_directory_config_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ConfigError, "is a directory"):
                load_config(config_path=temp_dir)

    def test_load_config_rejects_missing_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = str(Path(temp_dir) / "absent.json")
            with self.assertRaisesRegex(ConfigError, "not found"):
                load_config(config_path=missing)

    def test_load_config_rejects_invalid_json_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with self.assertRaisesRegex(ConfigError, "Invalid JSON"):
                load_config(config_path=temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_config_accepts_valid_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"model": "file-model", "max_iterations": 15}, f)
            temp_path = f.name

        try:
            config = load_config(config_path=temp_path)
            self.assertEqual(config["model"], "file-model")
            self.assertEqual(config["max_iterations"], 15)
        finally:
            os.unlink(temp_path)


class ConfigDocumentationTests(unittest.TestCase):
    """Keep README.md aligned with the actual configuration surface."""

    @classmethod
    def setUpClass(cls):
        cls.readme = README_PATH.read_text(encoding="utf-8")

    def _configuration_section(self):
        start = self.readme.index("## Configuration")
        end = self.readme.index("\n## ", start + 1)
        return self.readme[start:end]

    def test_readme_documents_every_configuration_key_and_default(self):
        for key, value in DEFAULT_CONFIG.items():
            matching = [
                line for line in self.readme.splitlines()
                if f"`{key}`" in line
            ]
            self.assertTrue(matching, f"README does not document config key: {key}")
            self.assertTrue(
                any(f"`{value}`" in line for line in matching),
                f"README does not document the default value for {key}: {value}",
            )

    def test_readme_documents_every_environment_variable(self):
        for env_key, mapping in ENV_MAPPING.items():
            config_key = mapping if isinstance(mapping, str) else mapping[0]
            matching = [
                line for line in self.readme.splitlines()
                if f"`{env_key}`" in line
            ]
            self.assertTrue(
                matching, f"README does not document environment variable: {env_key}"
            )
            self.assertTrue(
                any(f"`{config_key}`" in line for line in matching),
                f"README does not map {env_key} to {config_key}",
            )

    def test_readme_does_not_document_unknown_environment_variables(self):
        documented = set(re.findall(r"MLX_[A-Z_]+", self.readme))
        self.assertTrue(documented)
        unknown = documented - set(ENV_MAPPING)
        self.assertEqual(unknown, set(), f"README documents unknown variables: {unknown}")

    def test_readme_documents_config_option_and_precedence(self):
        section = self._configuration_section()
        self.assertIn("`--config`", section)
        self.assertIn("```json", section)
        defaults_at = section.index("Built-in defaults")
        file_at = section.index("JSON configuration file")
        env_at = section.index("environment variables")
        self.assertLess(defaults_at, file_at)
        self.assertLess(file_at, env_at)

    def test_readme_documents_invalid_configuration_reporting(self):
        section = self._configuration_section()
        self.assertIn("Configuration error", section)
        self.assertIn("non-zero", section)

    def test_readme_json_example_is_valid_configuration(self):
        section = self._configuration_section()
        match = re.search(r"```json\n(.*?)```", section, re.DOTALL)
        self.assertIsNotNone(match, "README has no JSON configuration example")
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            self.fail(f"README JSON example is not valid JSON: {exc}")
        self.assertIsInstance(payload, dict)
        self.assertEqual(set(payload), set(DEFAULT_CONFIG))
        self.assertEqual(validate_config(payload), payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
