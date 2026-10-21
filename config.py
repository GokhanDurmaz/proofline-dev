"""Configuration loader for MLX GSQ agent."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


DEFAULT_CONFIG = {
    "api_url": "http://127.0.0.1:8000/v1/chat/completions",
    "model": "mlx-community/Mellum2-12B-A2.5B-Instruct-4bit",
    "max_iterations": 20,
    "finalization_iterations": 4,
    "max_source_files": 8,
    "max_file_chars": 30000,
    "max_output_chars": 12000,
    "temperature": 0,
    "max_tokens": 512,
}


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration values."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a dictionary")

    validated = {}

    if "api_url" in config:
        api_url = config["api_url"]
        if not isinstance(api_url, str) or not api_url.strip():
            raise ConfigError("api_url must be a non-empty string")
        if not (api_url.startswith("http://") or api_url.startswith("https://")):
            raise ConfigError("api_url must start with http:// or https://")
        validated["api_url"] = api_url.strip()

    if "model" in config:
        model = config["model"]
        if not isinstance(model, str) or not model.strip():
            raise ConfigError("model must be a non-empty string")
        validated["model"] = model.strip()

    if "max_iterations" in config:
        max_iterations = config["max_iterations"]
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise ConfigError("max_iterations must be an integer")
        if max_iterations < 1:
            raise ConfigError("max_iterations must be at least 1")
        validated["max_iterations"] = max_iterations

    if "finalization_iterations" in config:
        finalization_iterations = config["finalization_iterations"]
        if not isinstance(finalization_iterations, int) or isinstance(finalization_iterations, bool):
            raise ConfigError("finalization_iterations must be an integer")
        if finalization_iterations < 0:
            raise ConfigError("finalization_iterations must be non-negative")
        validated["finalization_iterations"] = finalization_iterations

    if "max_source_files" in config:
        max_source_files = config["max_source_files"]
        if not isinstance(max_source_files, int) or isinstance(max_source_files, bool):
            raise ConfigError("max_source_files must be an integer")
        if max_source_files < 1:
            raise ConfigError("max_source_files must be at least 1")
        validated["max_source_files"] = max_source_files

    if "max_file_chars" in config:
        max_file_chars = config["max_file_chars"]
        if not isinstance(max_file_chars, int) or isinstance(max_file_chars, bool):
            raise ConfigError("max_file_chars must be an integer")
        if max_file_chars < 1:
            raise ConfigError("max_file_chars must be at least 1")
        validated["max_file_chars"] = max_file_chars

    if "max_output_chars" in config:
        max_output_chars = config["max_output_chars"]
        if not isinstance(max_output_chars, int) or isinstance(max_output_chars, bool):
            raise ConfigError("max_output_chars must be an integer")
        if max_output_chars < 1:
            raise ConfigError("max_output_chars must be at least 1")
        validated["max_output_chars"] = max_output_chars

    if "temperature" in config:
        temperature = config["temperature"]
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ConfigError("temperature must be a number")
        if temperature < 0 or temperature > 2:
            raise ConfigError("temperature must be between 0 and 2")
        validated["temperature"] = temperature

    if "max_tokens" in config:
        max_tokens = config["max_tokens"]
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise ConfigError("max_tokens must be an integer")
        if max_tokens < 1:
            raise ConfigError("max_tokens must be at least 1")
        validated["max_tokens"] = max_tokens

    return validated


def load_config(config_path: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """Load configuration from file with validation and apply overrides."""
    config = DEFAULT_CONFIG.copy()

    # Try to load from file if specified
    if config_path:
        path = Path(config_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if path.suffix.lower() == '.json':
                        file_config = json.load(f)
                    else:
                        # Try JSON as default
                        content = f.read()
                        file_config = json.loads(content)
                if not isinstance(file_config, dict):
                    raise ConfigError("Configuration file must contain a JSON object")
                validated_file = validate_config(file_config)
                config.update(validated_file)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid JSON in config file: {e}")
            except ConfigError:
                raise
            except Exception as e:
                raise ConfigError(f"Error reading config file: {e}")
        else:
            raise ConfigError(f"Configuration file not found: {config_path}")

    # Load from environment variables
    env_mapping = {
        "MLX_API_URL": "api_url",
        "MLX_MODEL": "model",
        "MLX_MAX_ITERATIONS": ("max_iterations", int),
        "MLX_MAX_SOURCE_FILES": ("max_source_files", int),
        "MLX_MAX_TOKENS": ("max_tokens", int),
        "MLX_TEMPERATURE": ("temperature", float),
    }

    for env_key, mapping in env_mapping.items():
        if env_key in os.environ:
            env_value = os.environ[env_key]
            if isinstance(mapping, tuple):
                key, converter = mapping
                try:
                    converted = converter(env_value)
                    env_config = {key: converted}
                    validated_env = validate_config(env_config)
                    config.update(validated_env)
                except (ValueError, ConfigError) as e:
                    raise ConfigError(f"Invalid environment variable {env_key}: {e}")
            else:
                env_config = {mapping: env_value}
                validated_env = validate_config(env_config)
                config.update(validated_env)

    # Apply overrides
    if overrides:
        validated_overrides = validate_config(overrides)
        config.update(validated_overrides)

    return config
