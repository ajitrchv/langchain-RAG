"""User-configurable LLM settings (provider, model, Ollama URL).

Only non-secret values are persisted to disk. API keys are never written to a file;
they live in the Streamlit session (memory) or in environment variables.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

PROVIDERS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "ollama": "Ollama (local)",
}
DEFAULT_MODELS = {
    "openai": "gpt-5.2",
    "anthropic": "claude-sonnet-4-5",
    "ollama": "llama3.1",
}
DEFAULT_OLLAMA_URL = "http://localhost:11434"
ENV_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def default_settings() -> Dict[str, Any]:
    return {
        "provider": "openai",
        "models": dict(DEFAULT_MODELS),
        "ollama_base_url": DEFAULT_OLLAMA_URL,
    }


def validate_base_url(url: str) -> str:
    """Return a normalised Ollama URL. Plain http is only allowed for local hosts."""
    url = (url or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a valid URL such as http://localhost:11434.")
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
        raise ValueError("Use https:// for remote servers (http:// is only allowed for localhost).")
    return url


def load_settings() -> Dict[str, Any]:
    """Load saved settings, falling back to defaults for anything missing or invalid."""
    settings = default_settings()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if not isinstance(data, dict):
        return settings

    if data.get("provider") in PROVIDERS:
        settings["provider"] = data["provider"]
    saved_models = data.get("models")
    if isinstance(saved_models, dict):
        for provider in PROVIDERS:
            model = saved_models.get(provider)
            if isinstance(model, str) and model.strip():
                settings["models"][provider] = model.strip()
    try:
        settings["ollama_base_url"] = validate_base_url(data.get("ollama_base_url", ""))
    except ValueError:
        pass
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    """Persist non-secret settings atomically with owner-only permissions."""
    tmp_path = SETTINGS_FILE.with_suffix(".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(settings, f)
    os.replace(tmp_path, SETTINGS_FILE)


def list_ollama_models(base_url: str) -> List[str]:
    """Names of the models installed on the Ollama server (empty if unreachable)."""
    try:
        import ollama

        response = ollama.Client(host=base_url, timeout=5).list()
        return sorted(m.model for m in response.models if m.model)
    except Exception:
        return []


def key_source(provider: str, session_keys: Dict[str, str]) -> str:
    """Where the API key for a provider currently comes from (never the key itself)."""
    if session_keys.get(provider):
        return "session"
    if os.environ.get(ENV_KEYS.get(provider, "")):
        return "environment"
    return "none"


def build_llm_config(settings: Dict[str, Any], session_keys: Dict[str, str]) -> Dict[str, Any]:
    """Chat-model config handed to the backend. Contains the API key, so never log it."""
    provider = settings["provider"]
    config: Dict[str, Any] = {"provider": provider, "model": settings["models"][provider]}
    if provider == "ollama":
        config["base_url"] = settings["ollama_base_url"]
    elif session_keys.get(provider):
        config["api_key"] = session_keys[provider]
    return config
