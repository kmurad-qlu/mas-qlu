import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env(env_file: Optional[str] = None) -> None:
    """
    Load environment variables for OpenRouter/OpenAI configuration.
    Falls back to loading a local env.example if .env is not present.
    """
    if env_file:
        load_dotenv(env_file, override=False)
        return

    # Try project-root .env first
    root_env = Path(__file__).resolve().parents[3] / ".env"
    if root_env.exists():
        load_dotenv(root_env.as_posix(), override=False)

    # Try app-level .env as second option
    app_env = Path(__file__).resolve().parents[1] / ".env"
    if app_env.exists():
        load_dotenv(app_env.as_posix(), override=False)

    # If no .env, load env.example to document expected vars (no override)
    app_example = Path(__file__).resolve().parents[1] / "env.example"
    if app_example.exists():
        load_dotenv(app_example.as_posix(), override=False)


def get_openrouter_api_key() -> Optional[str]:
    """
    Returns the OpenRouter API key. If only OPENAI_API_KEY is set,
    returns that as a fallback.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    return os.getenv("OPENAI_API_KEY")


def get_openai_base_url() -> str:
    """
    Return the base URL; default to OpenRouter if unspecified.
    """
    return os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def get_openrouter_headers() -> dict:
    """
    Extra headers recommended by OpenRouter to identify your app.
    """
    referer = os.getenv("OPENROUTER_REFERER", "https://example.org/mas")
    title = os.getenv("OPENROUTER_TITLE", "MAS-Orchestrator")
    return {
        "HTTP-Referer": referer,
        "X-Title": title,
    }

