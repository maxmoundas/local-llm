import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Environment variable overrides (MODEL_REPO_ID, SERVER_PORT, etc.)
    _apply_env_overrides(cfg)
    return cfg


def local_model_enabled(cfg: dict[str, Any]) -> bool:
    """Whether the built-in local model should be created and loaded (off unless enabled)."""
    return bool((cfg.get("model") or {}).get("enabled", False))


def _apply_env_overrides(cfg: dict) -> None:
    env_map = {
        "LOCAL_MODEL_ENABLED": ("model", "enabled"),
        "MODEL_REPO_ID": ("model", "repo_id"),
        "MODEL_CACHE_DIR": ("model", "cache_dir"),
        "MODEL_QUANTIZATION": ("model", "quantization"),
        "MODEL_MAX_NEW_TOKENS": ("model", "max_new_tokens"),
        "MODEL_TEMPERATURE": ("model", "temperature"),
        "SERVER_HOST": ("server", "host"),
        "SERVER_PORT": ("server", "port"),
        "LOG_LEVEL": ("logging", "level"),
    }
    for env_key, (section, key) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            target = cfg.setdefault(section, {})
            # Preserve type of the original value
            original = target.get(key)
            if isinstance(original, bool) or key == "enabled":  # bool before int: bool is an int subclass
                target[key] = val.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(original, int):
                target[key] = int(val)
            elif isinstance(original, float):
                target[key] = float(val)
            else:
                target[key] = val
