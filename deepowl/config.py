from pathlib import Path
import yaml

CONFIG_DIR = Path.home() / ".deepowl"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "model": {
        "provider": "ollama",
        "name": "qwen3:latest",
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


def load_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(yaml.dump(DEFAULT_CONFIG, default_flow_style=False))
    with CONFIG_FILE.open() as f:
        return yaml.safe_load(f)


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
