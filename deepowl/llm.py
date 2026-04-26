"""
Unified LLM + embedding adapter.

Supported providers:
  ollama    — local Ollama server (default)
  openai    — OpenAI API  (requires OPENAI_API_KEY)
  anthropic — Anthropic API (requires ANTHROPIC_API_KEY; no native embeddings)
  groq      — Groq API (requires GROQ_API_KEY; no native embeddings)

Embedding is always separate from chat — set embedding_provider in config.
Anthropic and Groq don't offer embeddings; pair them with provider=ollama or openai
for embedding_provider.
"""

import os
import sys

# Suggested defaults per provider (used by `deepowl config set provider <name>`)
PROVIDER_DEFAULTS = {
    "ollama": {
        "name": "qwen3:latest",
        "embedding": "nomic-embed-text",
        "embedding_provider": "ollama",
    },
    "openai": {
        "name": "gpt-4o-mini",
        "embedding": "text-embedding-3-small",
        "embedding_provider": "openai",
    },
    "anthropic": {
        "name": "claude-haiku-4-5-20251001",
        "embedding": "nomic-embed-text",
        "embedding_provider": "ollama",
    },
    "groq": {
        "name": "llama-3.1-8b-instant",
        "embedding": "nomic-embed-text",
        "embedding_provider": "ollama",
    },
}


# ── Chat ──────────────────────────────────────────────────────────────────────

def call_llm(provider: str, model: str, system: str, user: str) -> str:
    """Stream LLM response to stdout. Returns full text."""
    if provider == "ollama":
        return _ollama_chat(model, system, user)
    elif provider == "openai":
        return _openai_chat(model, system, user)
    elif provider == "anthropic":
        return _anthropic_chat(model, system, user)
    elif provider == "groq":
        return _groq_chat(model, system, user)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Choose: ollama, openai, anthropic, groq")


def _ollama_chat(model: str, system: str, user: str) -> str:
    import ollama
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        result = ""
        for chunk in stream:
            piece = chunk["message"]["content"]
            result += piece
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        return result
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}\nIs Ollama running? Try: ollama serve") from e


def _openai_chat(model: str, system: str, user: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install 'deepowl-learn[openai]'")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=True,
    )
    result = ""
    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""
        result += piece
        sys.stdout.write(piece)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return result


def _anthropic_chat(model: str, system: str, user: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install 'deepowl-learn[anthropic]'")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)
    result = ""
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for piece in stream.text_stream:
            result += piece
            sys.stdout.write(piece)
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return result


def _groq_chat(model: str, system: str, user: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install 'deepowl-learn[openai]'")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=True,
    )
    result = ""
    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""
        result += piece
        sys.stdout.write(piece)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return result


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embedding(provider: str, model: str, text: str) -> list[float]:
    if provider == "ollama":
        return _ollama_embed(model, text)
    elif provider == "openai":
        return _openai_embed(model, text)
    else:
        raise ValueError(
            f"Embedding not supported for provider: {provider!r}. "
            "Use 'ollama' or 'openai' as embedding_provider."
        )


def _ollama_embed(model: str, text: str) -> list[float]:
    import ollama
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def _openai_embed(model: str, text: str) -> list[float]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install 'deepowl-learn[openai]'")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
