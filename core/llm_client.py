"""
llm_client.py - LLM Backend Client for DevastatorAI

Handles communication with three backends:
  - Ollama      (local, no API key required)
  - OpenRouter  (cloud, requires OPENROUTER_API_KEY)
  - Anthropic   (cloud, requires ANTHROPIC_API_KEY — used for Opus synthesis in Full Ops)

Backend selection order (auto mode):
  1. If ANTHROPIC_API_KEY is set and the model name starts with "claude-" → Anthropic
  2. If OPENROUTER_API_KEY is set → OpenRouter
  3. Fallback → Ollama

Callers can override auto-detection via the backend parameter in send_prompt().

Config (read from .env):
    OLLAMA_URL          — default: http://localhost:11434
    OPENROUTER_API_KEY  — leave blank to use Ollama
    ANTHROPIC_API_KEY   — enables Anthropic backend (claude-opus-4-5 for Full Ops synthesis)
    DEFAULT_MODEL       — model name, default: qwen2.5:7b

Public API:
    send_prompt(agent_config, prompt, system_prompt=None, backend=None) -> str
    resolve_model(agent_config) -> str
    check_ollama() -> bool
    get_ollama_models() -> list
    clear_history(agent_name=None)
"""

import json
import os
import re
import urllib.request
import urllib.error
import logging
from pathlib import Path

logger = logging.getLogger("llm_client")


# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / ".env.example"
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


_env = _load_env()

OLLAMA_URL          = os.environ.get("OLLAMA_URL")          or _env.get("OLLAMA_URL", "http://localhost:11434")
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY")  or _env.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY")   or _env.get("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL       = os.environ.get("DEFAULT_MODEL")       or _env.get("DEFAULT_MODEL", "qwen2.5:7b")

# In-process conversation history per agent
_history: dict = {}
MAX_HISTORY_PAIRS = 10


# ---------------------------------------------------------------------------
# Thinking-block stripper (Qwen 3.5 / models that emit internal reasoning)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove 'Thinking... ...done thinking.' blocks from model output."""
    if not text:
        return text
    return re.sub(r'Thinking\.\.\..*?\.\.\.done thinking\.\s*', '', text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def get_history(agent_name: str) -> list:
    return _history.setdefault(agent_name, [])


def append_history(agent_name: str, role: str, content: str):
    history = get_history(agent_name)
    history.append({"role": role, "content": content})
    max_messages = MAX_HISTORY_PAIRS * 2
    if len(history) > max_messages:
        _history[agent_name] = history[-max_messages:]


def clear_history(agent_name: str = None):
    """Clear history for one agent, or all agents if name is None."""
    if agent_name:
        _history.pop(agent_name, None)
    else:
        _history.clear()


# ---------------------------------------------------------------------------
# HTTP helper (used by Ollama and OpenRouter)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict = None, timeout: int = 1800) -> dict:
    """POST JSON to url, return parsed response dict. Raises on HTTP error."""
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Backend: Ollama
# ---------------------------------------------------------------------------

def _send_ollama(model: str, system_prompt: str, agent_name: str, user_message: str) -> str:
    """Send a message to a local Ollama instance."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_history(agent_name))
    messages.append({"role": "user", "content": user_message})

    try:
        data = _post_json(
            f"{OLLAMA_URL}/api/chat",
            {"model": model, "messages": messages, "stream": False, "think": False},
        )
        return _strip_thinking(data.get("message", {}).get("content", ""))
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach Ollama at {OLLAMA_URL}: {e}") from e


# ---------------------------------------------------------------------------
# Backend: OpenRouter
# ---------------------------------------------------------------------------

def _send_openrouter(model: str, system_prompt: str, agent_name: str, user_message: str) -> str:
    """Send a message to OpenRouter."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_history(agent_name))
    messages.append({"role": "user", "content": user_message})

    try:
        data = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"model": model, "messages": messages},
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://github.com/KSirys/DevastatorAI",
                "X-Title": "DevastatorAI",
            },
        )
        return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach OpenRouter: {e}") from e
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected OpenRouter response shape: {e}") from e


# ---------------------------------------------------------------------------
# Backend: Anthropic
# ---------------------------------------------------------------------------

def _send_anthropic(model: str, system_prompt: str, agent_name: str, user_message: str) -> str:
    """
    Send a message to the Anthropic API using the anthropic Python package.

    Anthropic's API takes `system` as a top-level parameter (not inside messages).
    Conversation history is included as user/assistant message pairs.

    Requires: pip install anthropic
    """
    try:
        import anthropic as anthropic_sdk
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic backend. "
            "Install it with: pip install anthropic"
        )

    client = anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build messages: history (user/assistant only) + current user message
    messages = list(get_history(agent_name))
    messages.append({"role": "user", "content": user_message})

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        return message.content[0].text
    except anthropic_sdk.APIConnectionError as e:
        raise ConnectionError(f"Cannot reach Anthropic API: {e}") from e
    except anthropic_sdk.AuthenticationError as e:
        raise ValueError(f"Anthropic API key is invalid: {e}") from e
    except anthropic_sdk.RateLimitError as e:
        raise RuntimeError(f"Anthropic rate limit reached: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _select_backend(model: str, backend: str | None) -> str:
    """
    Determine which backend to use.

    Priority when backend=None (auto):
      1. ANTHROPIC_API_KEY set AND model starts with "claude-" → anthropic
      2. OPENROUTER_API_KEY set → openrouter
      3. Fallback → ollama

    Explicit backend values ("anthropic", "openrouter", "ollama") bypass auto-detection.
    """
    if backend in ("anthropic", "openrouter", "ollama"):
        return backend

    # Auto-detect
    if ANTHROPIC_API_KEY and model.startswith("claude-"):
        return "anthropic"
    if OPENROUTER_API_KEY:
        return "openrouter"
    return "ollama"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def resolve_model(agent_config: dict) -> str:
    """
    Determine which model to use for this agent.
    The placeholder '${DEFAULT_MODEL}' is treated the same as unset.
    Priority: agent_config["model"] > DEFAULT_MODEL env var > qwen2.5:7b
    """
    model = agent_config.get("model", "")
    if not model or model == "${DEFAULT_MODEL}":
        return DEFAULT_MODEL
    return model


def send_prompt(
    agent_config: dict,
    prompt: str,
    system_prompt: str = None,
    backend: str = None,
) -> str:
    """
    Send a prompt to the appropriate LLM backend and return the response string.

    Args:
        agent_config:  agent dict loaded from agents/*.json
        prompt:        the user message to send
        system_prompt: pre-built system prompt (e.g. from rules_engine);
                       if None, the agent's system_prompt field is used
        backend:       force a specific backend: "anthropic", "openrouter", or "ollama"
                       if None, auto-detection applies (see _select_backend)

    Returns:
        str — the model's response text

    Raises:
        ConnectionError — if the backend is unreachable
        ValueError      — if the response is malformed or the API key is invalid
        ImportError     — if the anthropic package is not installed and backend="anthropic"
    """
    agent_name = agent_config.get("name", "unknown")
    model = resolve_model(agent_config)
    sys_prompt = system_prompt or agent_config.get("system_prompt", f"You are {agent_name}.")

    selected_backend = _select_backend(model, backend)
    logger.info("[%s] Backend: %s | Model: %s", agent_name, selected_backend, model)

    if selected_backend == "anthropic":
        response = _send_anthropic(model, sys_prompt, agent_name, prompt)
    elif selected_backend == "openrouter":
        response = _send_openrouter(model, sys_prompt, agent_name, prompt)
    else:
        response = _send_ollama(model, sys_prompt, agent_name, prompt)

    append_history(agent_name, "user", prompt)
    append_history(agent_name, "assistant", response)

    return response


def check_ollama() -> bool:
    """Return True if Ollama is reachable at OLLAMA_URL."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_ollama_models() -> list:
    """Return list of locally available model names from Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.error("Failed to fetch Ollama models: %s", e)
        return []


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== LLM Client Self-Test ===\n")
    print(f"OLLAMA_URL:         {OLLAMA_URL}")
    print(f"OPENROUTER_API_KEY: {'set' if OPENROUTER_API_KEY else 'not set'}")
    print(f"ANTHROPIC_API_KEY:  {'set' if ANTHROPIC_API_KEY else 'not set'}")
    print(f"DEFAULT_MODEL:      {DEFAULT_MODEL}")

    # Show which backend would be selected for a standard model
    auto = _select_backend(DEFAULT_MODEL, None)
    claude = _select_backend("claude-opus-4-5", None)
    print(f"Auto backend ({DEFAULT_MODEL}): {auto}")
    print(f"Auto backend (claude-opus-4-5):  {claude}")

    if not OPENROUTER_API_KEY and not ANTHROPIC_API_KEY:
        ok = check_ollama()
        print(f"\nOllama reachable: {ok}")
        if ok:
            print(f"Available models: {get_ollama_models()}")

    test_agent = {
        "name": "Charlie",
        "model": "${DEFAULT_MODEL}",
        "system_prompt": "You are Charlie, a coding agent. Keep answers to one sentence.",
    }

    print("\nSending test prompt...")
    try:
        response = send_prompt(test_agent, "Say hello in one sentence.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== Done ===")
