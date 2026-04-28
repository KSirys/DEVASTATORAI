"""
chief.py - Chief of Staff Routing Engine for DevastatorAI

Classifies user input and routes it to the correct agent.
Classification runs keyword matching first; falls back to LLM if ambiguous.

Operation modes (set OPERATION_MODE in .env):

  solo     — one agent handles the task, local Ollama only
             fastest, free, works offline
  team     — Chief classifies, routes to 1-3 agents sequentially
             uses OpenRouter free tier if API key is set, else Ollama
  fullops  — all relevant agents run in parallel using the best available model
             prints estimated cost before running, requires confirmation

Usage:
    python core/chief.py --prompt "Research the best Python web frameworks"
    python core/chief.py --prompt "Write a blog post about AI" --mode team
    python core/chief.py --prompt "Review this code for bugs" --mode fullops

Outputs JSON to stdout:
    {
        "response": "...",
        "agent": "Charlie",
        "mode": "solo",
        "elapsed_seconds": 4.2
    }
"""

import os
import sys
import json
import time
import logging
import argparse
import concurrent.futures
from pathlib import Path

logger = logging.getLogger("chief")


# ---------------------------------------------------------------------------
# Bootstrap: resolve project root and load env before other imports
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

def _load_env():
    env_path = _ROOT / ".env"
    if not env_path.exists():
        env_path = _ROOT / ".env.example"
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
OPERATION_MODE = os.environ.get("OPERATION_MODE") or _env.get("OPERATION_MODE", "solo")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or _env.get("OPENROUTER_API_KEY", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL") or _env.get("DEFAULT_MODEL", "qwen2.5:7b")


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

AGENTS = {
    "rachel": {
        "name": "Rachel",
        "role": "Research Agent",
        "description": "Searches and summarizes information",
        "model": "${DEFAULT_MODEL}",
        "system_prompt": (
            "You are Rachel, a research agent. Find, analyze, and summarize information "
            "clearly and accurately. Cite your reasoning. Be concise but thorough."
        ),
    },
    "winter": {
        "name": "Winter",
        "role": "Writing Agent",
        "description": "Drafts content and documents",
        "model": "${DEFAULT_MODEL}",
        "system_prompt": (
            "You are Winter, a writing agent. Draft clear, compelling, well-structured content. "
            "Adapt tone to the task. Always deliver complete drafts."
        ),
    },
    "charlie": {
        "name": "Charlie",
        "role": "Coding Agent",
        "description": "Writes and reviews code",
        "model": "${DEFAULT_MODEL}",
        "system_prompt": (
            "You are Charlie, a coding agent. Write clean, working code and review existing code "
            "for bugs, security issues, and improvements. Prefer simple, readable solutions."
        ),
    },
    "sentinel": {
        "name": "Sentinel",
        "role": "Security Agent",
        "description": "Monitors system processes and network activity",
        "model": "${DEFAULT_MODEL}",
        "system_prompt": (
            "You are Sentinel, a security agent. Analyze system data, process lists, and network "
            "activity for anomalies or threats. Report findings with severity: LOW/MEDIUM/HIGH/CRITICAL."
        ),
    },
}

# Models to use per agent in fullops mode (best available via OpenRouter)
FULLOPS_MODELS = {
    "rachel": "anthropic/claude-3-haiku",
    "winter": "anthropic/claude-3-haiku",
    "charlie": "anthropic/claude-3-5-sonnet",
    "sentinel": "anthropic/claude-3-haiku",
}

# Estimated cost per 1M output tokens (USD) — rough figures for cost preview
FULLOPS_COST_PER_1M = {
    "anthropic/claude-3-haiku": 1.25,
    "anthropic/claude-3-5-sonnet": 15.00,
}


# ---------------------------------------------------------------------------
# Keyword classifier
# ---------------------------------------------------------------------------

_KEYWORD_MAP = {
    "rachel": [
        "research", "search", "find", "look up", "investigate", "summarize",
        "what is", "who is", "explain", "information about", "tell me about",
        "latest news", "current events", "facts about", "history of",
    ],
    "winter": [
        "write", "draft", "create", "compose", "document", "blog", "article",
        "email", "report", "essay", "outline", "rewrite", "edit", "proofread",
        "copy", "letter", "post", "describe", "narrative",
    ],
    "charlie": [
        "code", "debug", "fix", "program", "script", "function", "class",
        "implement", "build", "develop", "refactor", "review code", "test",
        "bug", "error", "exception", "syntax", "algorithm", "api", "database",
    ],
    "sentinel": [
        "security", "monitor", "scan", "threat", "anomaly", "process",
        "network", "intrusion", "malware", "vulnerability", "suspicious",
        "alert", "breach", "attack", "firewall", "audit", "port",
    ],
}


def classify_by_keywords(prompt: str) -> tuple[str | None, float]:
    """
    Score each agent by keyword matches in the prompt.
    Returns (agent_key, confidence) or (None, 0) if ambiguous.
    Confidence is 1.0 if one agent clearly leads, lower if tied.
    """
    prompt_lower = prompt.lower()
    scores = {agent: 0 for agent in _KEYWORD_MAP}

    for agent, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in prompt_lower:
                scores[agent] += 1

    best_agent = max(scores, key=scores.get)
    best_score = scores[best_agent]

    if best_score == 0:
        return None, 0.0

    # Check for a tie
    tied = [a for a, s in scores.items() if s == best_score]
    if len(tied) > 1:
        return None, 0.5  # Ambiguous — fall through to LLM

    return best_agent, 1.0


def classify_with_llm(prompt: str) -> str:
    """
    Ask the LLM to classify which agent should handle the prompt.
    Returns one of: rachel, winter, charlie, sentinel.
    """
    from core.llm_client import send_prompt

    classification_agent = {
        "name": "Chief",
        "model": "${DEFAULT_MODEL}",
        "system_prompt": (
            "You are a task router. Given a user prompt, respond with exactly one word: "
            "the agent that should handle it.\n"
            "Agents:\n"
            "  rachel   — research, summarization, fact-finding\n"
            "  winter   — writing, drafting, editing\n"
            "  charlie  — coding, debugging, technical review\n"
            "  sentinel — security, monitoring, threat analysis\n"
            "Respond with only the agent name, nothing else."
        ),
    }

    try:
        result = send_prompt(classification_agent, prompt)
        choice = result.strip().lower().split()[0]
        if choice in AGENTS:
            return choice
    except Exception as e:
        logger.error("LLM classification failed: %s", e)

    return "rachel"  # Safe default


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def _load_agent_config(agent_key: str, override_model: str = None) -> dict:
    config = dict(AGENTS[agent_key])
    if override_model:
        config["model"] = override_model
    return config


def run_solo(prompt: str) -> dict:
    """
    Solo mode: classify prompt, send to one agent via local Ollama.
    No OpenRouter required.
    """
    from core.llm_client import send_prompt
    from core.rules_engine import build_system_prompt, scan_for_injections, wrap_untrusted_content, validate_response

    agent_key, confidence = classify_by_keywords(prompt)
    if not agent_key or confidence < 1.0:
        agent_key = classify_with_llm(prompt)

    agent_config = _load_agent_config(agent_key)
    sys_prompt = build_system_prompt(agent_config["name"])

    scan = scan_for_injections(prompt)
    safe_prompt = wrap_untrusted_content(prompt) if scan["is_suspicious"] else prompt
    if scan["is_suspicious"]:
        logger.warning("Injection patterns in user prompt: %s", scan["threats_found"])

    start = time.time()
    response = send_prompt(agent_config, safe_prompt, system_prompt=sys_prompt)
    elapsed = round(time.time() - start, 2)

    validation = validate_response(response)

    return {
        "response": validation["sanitized_response"],
        "agent": agent_config["name"],
        "mode": "solo",
        "is_safe": validation["is_safe"],
        "violations": validation["violations"],
        "elapsed_seconds": elapsed,
    }


def run_team(prompt: str) -> dict:
    """
    Team mode: Chief classifies, routes to 1-3 agents sequentially.
    Uses OpenRouter if API key is set, otherwise Ollama.
    """
    from core.llm_client import send_prompt
    from core.rules_engine import build_system_prompt, validate_response

    agent_key, confidence = classify_by_keywords(prompt)
    if not agent_key or confidence < 1.0:
        agent_key = classify_with_llm(prompt)

    # In team mode, include Rachel as co-analyst for research + other tasks
    team_agents = [agent_key]
    if agent_key != "rachel" and agent_key in ("winter", "charlie"):
        team_agents.insert(0, "rachel")

    results = []
    total_start = time.time()

    for key in team_agents:
        agent_config = _load_agent_config(key)
        sys_prompt = build_system_prompt(agent_config["name"])

        task_prompt = prompt
        if len(team_agents) > 1 and key == team_agents[-1]:
            # Give the primary agent the research context
            prior = results[-1]["response"] if results else ""
            if prior:
                task_prompt = (
                    f"{prompt}\n\n"
                    f"[Context from {AGENTS[team_agents[0]]['name']}]: {prior[:1500]}"
                )

        start = time.time()
        response = send_prompt(agent_config, task_prompt, system_prompt=sys_prompt)
        elapsed = round(time.time() - start, 2)
        validation = validate_response(response)
        results.append({
            "agent": agent_config["name"],
            "response": validation["sanitized_response"],
            "elapsed_seconds": elapsed,
            "is_safe": validation["is_safe"],
        })

    primary = results[-1]
    total_elapsed = round(time.time() - total_start, 2)

    return {
        "response": primary["response"],
        "agent": primary["agent"],
        "mode": "team",
        "team_results": results,
        "is_safe": primary["is_safe"],
        "violations": [],
        "elapsed_seconds": total_elapsed,
    }


def _estimate_cost(agents: list, prompt: str) -> str:
    """Estimate fullops cost based on prompt length and model pricing."""
    est_input_tokens = len(prompt.split()) * 1.4
    est_output_tokens = 500  # rough per-agent estimate
    lines = ["Estimated cost (fullops):"]
    total = 0.0
    for key in agents:
        model = FULLOPS_MODELS.get(key, DEFAULT_MODEL)
        cost_per_1m = FULLOPS_COST_PER_1M.get(model, 1.0)
        agent_cost = (est_output_tokens / 1_000_000) * cost_per_1m
        total += agent_cost
        lines.append(f"  {AGENTS[key]['name']:12s} ({model}): ~${agent_cost:.4f}")
    lines.append(f"  {'TOTAL':12s}                  ~${total:.4f}")
    return "\n".join(lines)


def run_fullops(prompt: str) -> dict:
    """
    Fullops mode: all agents run in parallel using the best available models.
    Prints cost estimate and requires confirmation before proceeding.
    Requires OPENROUTER_API_KEY.
    """
    from core.llm_client import send_prompt
    from core.rules_engine import build_system_prompt, validate_response

    all_agents = list(AGENTS.keys())

    cost_preview = _estimate_cost(all_agents, prompt)
    print(f"\n{cost_preview}")
    print("\nProceed with fullops? [y/N] ", end="", flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = "n"

    if answer not in ("y", "yes"):
        return {
            "response": "Fullops cancelled by user.",
            "agent": "Chief",
            "mode": "fullops",
            "elapsed_seconds": 0,
        }

    def run_one(agent_key):
        config = _load_agent_config(agent_key, override_model=FULLOPS_MODELS.get(agent_key))
        sys_prompt = build_system_prompt(config["name"])
        t0 = time.time()
        response = send_prompt(config, prompt, system_prompt=sys_prompt)
        validation = validate_response(response)
        return {
            "agent": config["name"],
            "response": validation["sanitized_response"],
            "elapsed_seconds": round(time.time() - t0, 2),
            "is_safe": validation["is_safe"],
        }

    total_start = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_agents)) as pool:
        futures = {pool.submit(run_one, key): key for key in all_agents}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                key = futures[future]
                logger.error("Fullops agent %s failed: %s", key, e)
                results.append({"agent": AGENTS[key]["name"], "response": f"Error: {e}", "is_safe": False})

    total_elapsed = round(time.time() - total_start, 2)

    # Primary response = the agent that matched best by keyword
    agent_key, _ = classify_by_keywords(prompt)
    primary_name = AGENTS.get(agent_key, AGENTS["rachel"])["name"]
    primary = next((r for r in results if r["agent"] == primary_name), results[0])

    return {
        "response": primary["response"],
        "agent": primary["agent"],
        "mode": "fullops",
        "all_results": results,
        "elapsed_seconds": total_elapsed,
    }


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def run(prompt: str, mode: str = None) -> dict:
    """
    Route a prompt using the specified mode (or OPERATION_MODE from .env).

    Args:
        prompt: the user's input
        mode:   "solo", "team", or "fullops" — overrides OPERATION_MODE if set

    Returns:
        dict with keys: response, agent, mode, elapsed_seconds
    """
    active_mode = (mode or OPERATION_MODE).lower().strip()

    if active_mode == "fullops":
        return run_fullops(prompt)
    elif active_mode == "team":
        return run_team(prompt)
    else:
        return run_solo(prompt)


# ---------------------------------------------------------------------------
# CLI entry point (called by index.js via child process)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="DevastatorAI Chief of Staff")
    parser.add_argument("--prompt", required=True, help="User prompt to route")
    parser.add_argument("--mode", default=None, help="Operation mode: solo, team, fullops")
    args = parser.parse_args()

    result = run(args.prompt, mode=args.mode)
    print(json.dumps(result, ensure_ascii=False))
