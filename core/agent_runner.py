"""
agent_runner.py - Single-Agent CLI Runner for DevastatorAI

Loads an agent config from agents/, runs a prompt through the full
rules engine pipeline, sends to LLM, validates the response, and returns it.

Pipeline:
  1. Load agent config from agents/<name>.json
  2. Build system prompt via rules_engine (universal + per-agent rules)
  3. Scan prompt for injection patterns
  4. Send to LLM via llm_client (Ollama or OpenRouter)
  5. Validate response through rules_engine before returning

Usage:
    python core/agent_runner.py --agent rachel --prompt "Summarize neural networks"
    python core/agent_runner.py --agent charlie --prompt "Write a hello world in Rust"
    python core/agent_runner.py --agent sentinel --prompt "Explain these processes"
    python core/agent_runner.py --list
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.rules_engine import (
    build_system_prompt,
    scan_for_injections,
    wrap_untrusted_content,
    validate_response,
)
from core.llm_client import send_prompt, resolve_model

logger = logging.getLogger("agent_runner")

AGENTS_DIR = _ROOT / "agents"


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------

def list_agents() -> list[dict]:
    """Return all agent configs from agents/*.json."""
    agents = []
    if not AGENTS_DIR.exists():
        return agents
    for path in sorted(AGENTS_DIR.glob("*.json")):
        try:
            config = json.loads(path.read_text())
            config["_file"] = path.name
            agents.append(config)
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed agent file %s: %s", path.name, e)
    return agents


def load_agent(name: str) -> dict | None:
    """
    Find an agent by name (case-insensitive) or by filename stem.
    Returns the agent config dict, or None if not found.
    """
    name_lower = name.lower()
    for agent in list_agents():
        if agent.get("name", "").lower() == name_lower:
            return agent
        stem = agent.get("_file", "").replace(".json", "").lower()
        if stem == name_lower:
            return agent
    return None


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(agent_name: str, prompt: str) -> dict:
    """
    Run a prompt through the full pipeline for the named agent.

    Returns:
        dict:
            response      — the final response (validated, or block message)
            agent         — agent name that handled the request
            model         — model used
            backend       — "ollama" or "openrouter"
            is_safe       — True if response passed all rules checks
            violations    — list of violation descriptions (empty if safe)
            injection_warning — True if prompt contained suspicious patterns
    """
    from core.llm_client import OPENROUTER_API_KEY

    # Step 1: Load agent config
    agent_config = load_agent(agent_name)
    if not agent_config:
        available = [a.get("name", a.get("_file")) for a in list_agents()]
        return {
            "response": (
                f"Agent '{agent_name}' not found.\n"
                f"Available agents: {', '.join(available)}"
            ),
            "agent": None,
            "model": None,
            "backend": None,
            "is_safe": False,
            "violations": [f"Agent '{agent_name}' not found"],
            "injection_warning": False,
        }

    agent_display_name = agent_config.get("name", agent_name)
    model = resolve_model(agent_config)
    backend = "openrouter" if OPENROUTER_API_KEY else "ollama"

    # Step 2: Build system prompt with rules engine
    system_prompt = build_system_prompt(agent_display_name)

    # Step 3: Scan prompt for injection patterns
    scan = scan_for_injections(prompt)
    injection_warning = scan["is_suspicious"]
    if injection_warning:
        logger.warning(
            "[%s] Injection patterns in user prompt: %s",
            agent_display_name, scan["threats_found"]
        )
        safe_prompt = wrap_untrusted_content(prompt)
    else:
        safe_prompt = prompt

    # Step 4: Send to LLM
    try:
        raw_response = send_prompt(agent_config, safe_prompt, system_prompt=system_prompt)
    except ConnectionError as e:
        return {
            "response": f"[Connection Error] {e}",
            "agent": agent_display_name,
            "model": model,
            "backend": backend,
            "is_safe": False,
            "violations": [str(e)],
            "injection_warning": injection_warning,
        }
    except Exception as e:
        return {
            "response": f"[LLM Error] {e}",
            "agent": agent_display_name,
            "model": model,
            "backend": backend,
            "is_safe": False,
            "violations": [str(e)],
            "injection_warning": injection_warning,
        }

    # Step 5: Validate response before returning
    validation = validate_response(raw_response)
    if not validation["is_safe"]:
        logger.warning(
            "[%s] Response blocked: %s",
            agent_display_name, validation["violations"]
        )

    return {
        "response": validation["sanitized_response"],
        "agent": agent_display_name,
        "model": model,
        "backend": backend,
        "is_safe": validation["is_safe"],
        "violations": validation["violations"],
        "injection_warning": injection_warning,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_agents():
    agents = list_agents()
    if not agents:
        print("No agents found in agents/ folder.")
        return
    print(f"{'Agent':<20} {'Role':<30} {'File'}")
    print("-" * 65)
    for a in agents:
        print(f"{a.get('name', '?'):<20} {a.get('role', '?'):<30} {a.get('_file', '?')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="DevastatorAI Agent Runner")
    parser.add_argument("--agent", help="Agent name to use (e.g. rachel, charlie)")
    parser.add_argument("--prompt", help="Prompt to send to the agent")
    parser.add_argument("--list", action="store_true", help="List all available agents")
    parser.add_argument("--json", action="store_true", dest="output_json",
                        help="Output full result as JSON instead of just the response")
    args = parser.parse_args()

    if args.list:
        _print_agents()
        sys.exit(0)

    if not args.agent:
        parser.error("--agent is required (or use --list to see available agents)")
    if not args.prompt:
        parser.error("--prompt is required")

    result = run(args.agent, args.prompt)

    if args.output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["injection_warning"]:
            print("[Warning: injection patterns detected in prompt]", file=sys.stderr)
        print(result["response"])
        if not result["is_safe"]:
            print(f"\n[Blocked by rules engine: {result['violations']}]", file=sys.stderr)
