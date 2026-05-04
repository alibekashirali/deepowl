from pathlib import Path
import yaml

CONFIG_DIR = Path.home() / ".deepowl"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "model": {
        "provider": "ollama",
        "name": "qwen2.5:7b",
        "embedding": "nomic-embed-text",
        "embedding_provider": "ollama",
    },
    "storage": {
        "db_path": "~/.deepowl/deepowl.db",
        "chroma_path": "~/.deepowl/chroma",
    },
    "teaching": {
        "language": "auto",
        "style": "socratic",
        "session_length": 20,
        "spaced_repetition": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Fill missing keys in override from base (non-destructive)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(yaml.dump(DEFAULT_CONFIG, default_flow_style=False))
        return dict(DEFAULT_CONFIG)
    with CONFIG_FILE.open() as f:
        on_disk = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, on_disk)


def resolve_paths(config: dict) -> dict:
    """Expand ~ in storage paths and ensure directories exist."""
    storage = config["storage"]
    db_path = Path(storage["db_path"]).expanduser()
    chroma_path = Path(storage["chroma_path"]).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    chroma_path.mkdir(parents=True, exist_ok=True)
    config["storage"]["db_path"] = str(db_path)
    config["storage"]["chroma_path"] = str(chroma_path)
    return config
