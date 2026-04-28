"""
chief.py - Chief of Staff Routing Engine for DevastatorAI

Classifies user input and routes it to the correct agent(s).
Classification uses keyword matching first, LLM fallback if ambiguous.

Operation modes (set OPERATION_MODE in .env):

  solo     — one agent handles the full task, local Ollama only
             fastest, free, works completely offline

  team     — Chief classifies intent, builds a 2-3 agent sequential pipeline
             each agent's output becomes context for the next
             Chief reviews all output and returns a polished final response
             uses OpenRouter free tier if OPENROUTER_API_KEY is set, else Ollama

  fullops  — Chief determines relevant agents, runs them in PARALLEL
             cost is estimated and confirmed before anything runs
             Chief synthesizes all outputs into one final response
             uses best available models via OpenRouter
             if ANTHROPIC_API_KEY is set, Chief synthesis uses claude-opus-4-5

Usage:
    python core/chief.py --prompt "Research the best Python web frameworks"
    python core/chief.py --prompt "Write a blog post about AI" --mode team
    python core/chief.py --prompt "Review this architecture" --mode fullops

Outputs JSON to stdout:
    {
        "response": "...",
        "agent": "Winter",
        "mode": "team",
        "elapsed_seconds": 12.4
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
# Bootstrap: resolve project root and load .env before other imports
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
OPERATION_MODE       = os.environ.get("OPERATION_MODE")       or _env.get("OPERATION_MODE", "solo")
OPENROUTER_API_KEY   = os.environ.get("OPENROUTER_API_KEY")   or _env.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY")    or _env.get("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL        = os.environ.get("DEFAULT_MODEL")        or _env.get("DEFAULT_MODEL", "qwen2.5:7b")


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

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Team mode: free OpenRouter models when OPENROUTER_API_KEY is set
TEAM_MODEL_OPENROUTER = "nvidia/llama-3.1-nemotron-70b-instruct"

# Full Ops worker models (run in parallel)
FULLOPS_WORKER_MODELS = {
    "rachel":   "nvidia/llama-3.1-nemotron-70b-instruct",
    "winter":   "nvidia/llama-3.1-nemotron-70b-instruct",
    "charlie":  "deepseek/deepseek-r1",
    "sentinel": "nvidia/llama-3.1-nemotron-70b-instruct",
}

# Full Ops synthesis model (Chief final step)
FULLOPS_SYNTHESIS_OPENROUTER  = "nvidia/llama-3.1-nemotron-70b-instruct"
FULLOPS_SYNTHESIS_ANTHROPIC   = "claude-opus-4-5"

# Estimated cost per 1M output tokens (USD)
FULLOPS_COST_PER_1M = {
    "nvidia/llama-3.1-nemotron-70b-instruct": 0.00,   # free tier
    "deepseek/deepseek-r1":                   2.19,
    "claude-opus-4-5":                        75.00,
}


# ---------------------------------------------------------------------------
# Team mode pipeline definitions
#
# Each pipeline is an ordered list of (agent_key, role_description) tuples.
# Role descriptions are printed to the user and passed as context.
# ---------------------------------------------------------------------------

TEAM_PIPELINES = {
    "rachel": [
        ("rachel", "Research and gather all relevant information on the topic"),
        ("winter", "Write a clear, structured summary of the research findings"),
    ],
    "winter": [
        ("rachel", "Research supporting facts, statistics, and source material"),
        ("winter", "Draft the final content using the research as a foundation"),
    ],
    "charlie": [
        ("charlie", "Write the code or technical implementation"),
        ("rachel", "Review the solution for accuracy, correctness, and best practices"),
    ],
    "sentinel": [
        ("sentinel", "Perform full security analysis and threat assessment"),
    ],
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
    Score each agent by keyword matches.
    Returns (agent_key, confidence): confidence is 1.0 if unambiguous, 0.5 if tied, 0.0 if no match.
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

    tied = [a for a, s in scores.items() if s == best_score]
    if len(tied) > 1:
        return None, 0.5

    return best_agent, 1.0


def classify_with_llm(prompt: str) -> str:
    """Ask the LLM to classify which agent should handle the prompt."""
    from core.llm_client import send_prompt

    classifier_config = {
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
        result = send_prompt(classifier_config, prompt)
        choice = result.strip().lower().split()[0]
        if choice in AGENTS:
            return choice
    except Exception as e:
        logger.error("LLM classification failed: %s", e)

    return "rachel"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _agent_config(agent_key: str, model_override: str = None) -> dict:
    """Return a copy of the agent config, optionally with model overridden."""
    config = dict(AGENTS[agent_key])
    if model_override:
        config["model"] = model_override
    return config


def _team_model() -> str | None:
    """Return the model to use for team mode, or None to use DEFAULT_MODEL via Ollama."""
    return TEAM_MODEL_OPENROUTER if OPENROUTER_API_KEY else None


def _resolve_synthesis_backend() -> tuple[str, str]:
    """
    Determine the Chief synthesis model and backend for fullops.
    Returns (model_name, backend_hint) where backend_hint is "anthropic", "openrouter", or "ollama".
    """
    if ANTHROPIC_API_KEY:
        return FULLOPS_SYNTHESIS_ANTHROPIC, "anthropic"
    if OPENROUTER_API_KEY:
        return FULLOPS_SYNTHESIS_OPENROUTER, "openrouter"
    return DEFAULT_MODEL, "ollama"


# ---------------------------------------------------------------------------
# Solo mode
# ---------------------------------------------------------------------------

def run_solo(prompt: str) -> dict:
    """
    Solo mode: classify prompt, send to one agent via local Ollama.
    No API key required.
    """
    from core.llm_client import send_prompt
    from core.rules_engine import build_system_prompt, scan_for_injections, wrap_untrusted_content, validate_response

    agent_key, confidence = classify_by_keywords(prompt)
    if not agent_key or confidence < 1.0:
        agent_key = classify_with_llm(prompt)

    config = _agent_config(agent_key)
    sys_prompt = build_system_prompt(config["name"])

    scan = scan_for_injections(prompt)
    safe_prompt = wrap_untrusted_content(prompt) if scan["is_suspicious"] else prompt
    if scan["is_suspicious"]:
        logger.warning("Injection patterns in user prompt: %s", scan["threats_found"])

    t0 = time.time()
    response = send_prompt(config, safe_prompt, system_prompt=sys_prompt)
    elapsed = round(time.time() - t0, 2)

    validation = validate_response(response)

    return {
        "response": validation["sanitized_response"],
        "agent": config["name"],
        "mode": "solo",
        "is_safe": validation["is_safe"],
        "violations": validation["violations"],
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Team mode
# ---------------------------------------------------------------------------

def _chief_team_review(original_prompt: str, pipeline_log: list[dict]) -> str:
    """
    Chief's final review step in team mode.
    Reads all agent contributions and returns a polished, cohesive final response.
    """
    from core.llm_client import send_prompt

    contributions = "\n\n".join(
        f"[{entry['agent']} — {entry['role']}]\n{entry['response']}"
        for entry in pipeline_log
    )

    synthesis_prompt = (
        f"Original request: {original_prompt}\n\n"
        f"Your team has completed their work. Here are their contributions:\n\n"
        f"{contributions}\n\n"
        "Review everything above. Deliver a single, polished final response to the user. "
        "Integrate the best elements from all contributions. Do not explain the process — "
        "just give the user the best possible answer."
    )

    chief_config = {
        "name": "Chief",
        "model": _team_model() or "${DEFAULT_MODEL}",
        "system_prompt": (
            "You are the Chief of Staff — an experienced orchestrator who reviews your team's work "
            "and delivers the final, cohesive response to the user. You synthesize, polish, and "
            "present. You do not repeat the process — you deliver the result."
        ),
    }

    try:
        return send_prompt(chief_config, synthesis_prompt)
    except Exception as e:
        logger.error("Chief team review failed: %s", e)
        # Fall back to returning the last agent's response
        return pipeline_log[-1]["response"] if pipeline_log else "No response generated."


def run_team(prompt: str) -> dict:
    """
    Team mode: Chief builds a sequential agent pipeline, runs it, then reviews.

    Pipeline flow (example — research task):
        Rachel (research) → Winter (write summary) → Chief review → user

    Pipeline flow (example — coding task):
        Charlie (write code) → Rachel (review accuracy) → Chief review → user

    Each agent's output becomes context for the next agent.
    Pipeline is printed before execution begins.
    """
    from core.llm_client import send_prompt
    from core.rules_engine import build_system_prompt, scan_for_injections, wrap_untrusted_content, validate_response

    # Classify primary intent
    agent_key, confidence = classify_by_keywords(prompt)
    if not agent_key or confidence < 1.0:
        agent_key = classify_with_llm(prompt)

    pipeline = TEAM_PIPELINES.get(agent_key, TEAM_PIPELINES["rachel"])

    # Print pipeline before running
    agent_names = [AGENTS[key]["name"] for key, _ in pipeline]
    pipeline_display = " → ".join(agent_names) + " → Chief review"
    print(f"\n  [Team] Pipeline: {pipeline_display}", flush=True)
    for i, (key, role) in enumerate(pipeline, 1):
        print(f"    {i}. {AGENTS[key]['name']}: {role}", flush=True)
    print(f"    {len(pipeline) + 1}. Chief: Final review and synthesis", flush=True)
    print("", flush=True)

    # Scan prompt for injections
    scan = scan_for_injections(prompt)
    if scan["is_suspicious"]:
        logger.warning("Injection patterns in user prompt: %s", scan["threats_found"])

    pipeline_log = []
    total_start = time.time()
    team_model = _team_model()

    for step_num, (key, role_desc) in enumerate(pipeline):
        config = _agent_config(key, model_override=team_model)
        sys_prompt = build_system_prompt(config["name"])

        # Build context-aware prompt for this step
        if step_num == 0:
            # First agent gets the raw user prompt
            step_prompt = wrap_untrusted_content(prompt) if scan["is_suspicious"] else prompt
        else:
            # Subsequent agents get prior outputs as context
            prior_context = "\n\n".join(
                f"[{entry['agent']} — {entry['role']}]\n{entry['response']}"
                for entry in pipeline_log
            )
            step_prompt = (
                f"Original request: {prompt}\n\n"
                f"Previous team output:\n{prior_context}\n\n"
                f"Your task: {role_desc}. Build on the work above. "
                f"Do not repeat what was already done — continue from where they left off."
            )

        print(f"  [{step_num + 1}/{len(pipeline) + 1}] Running {config['name']}...", flush=True)
        t0 = time.time()

        try:
            response = send_prompt(config, step_prompt, system_prompt=sys_prompt)
        except Exception as e:
            response = f"[{config['name']} encountered an error: {e}]"
            logger.error("Team step %s failed: %s", key, e)

        elapsed = round(time.time() - t0, 2)
        validation = validate_response(response)

        pipeline_log.append({
            "agent": config["name"],
            "role": role_desc,
            "response": validation["sanitized_response"],
            "elapsed_seconds": elapsed,
            "is_safe": validation["is_safe"],
        })

        print(f"      Done ({elapsed}s)", flush=True)

    # Chief final review
    print(f"  [{len(pipeline) + 1}/{len(pipeline) + 1}] Chief reviewing and synthesizing...", flush=True)
    t0 = time.time()
    final_response = _chief_team_review(prompt, pipeline_log)
    chief_elapsed = round(time.time() - t0, 2)
    print(f"      Done ({chief_elapsed}s)", flush=True)

    total_elapsed = round(time.time() - total_start, 2)

    return {
        "response": final_response,
        "agent": "Chief of Staff",
        "mode": "team",
        "pipeline": pipeline_display,
        "pipeline_log": pipeline_log,
        "is_safe": True,
        "violations": [],
        "elapsed_seconds": total_elapsed,
    }


# ---------------------------------------------------------------------------
# Full Ops mode
# ---------------------------------------------------------------------------

def _determine_relevant_agents(prompt: str) -> list[str]:
    """
    Determine which agents are most relevant to the task.
    Returns 2-4 agent keys, always at least 2.
    """
    prompt_lower = prompt.lower()
    scores = {agent: 0 for agent in _KEYWORD_MAP}

    for agent, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in prompt_lower:
                scores[agent] += 1

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Include any agent with score > 0, plus always top 2
    selected = [key for key, score in ranked if score > 0]
    if len(selected) < 2:
        selected = [key for key, _ in ranked[:2]]

    return selected


def _estimate_fullops_cost(agent_keys: list[str], prompt: str) -> tuple[str, float]:
    """
    Build the cost estimate string and return (display_string, total_usd).
    Accounts for worker agents + synthesis step.
    """
    est_output_tokens = 600  # rough per-agent estimate

    synthesis_model, synthesis_backend = _resolve_synthesis_backend()

    lines = []
    total = 0.0

    # Worker agents
    for key in agent_keys:
        model = FULLOPS_WORKER_MODELS.get(key, DEFAULT_MODEL) if OPENROUTER_API_KEY else DEFAULT_MODEL
        cost_per_1m = FULLOPS_COST_PER_1M.get(model, 0.00)
        agent_cost = (est_output_tokens / 1_000_000) * cost_per_1m
        total += agent_cost
        lines.append(f"    {AGENTS[key]['name']:<14} {model}  ~${agent_cost:.4f}")

    # Synthesis step
    synthesis_cost = (est_output_tokens / 1_000_000) * FULLOPS_COST_PER_1M.get(synthesis_model, 0.00)
    total += synthesis_cost
    lines.append(f"    {'Chief synthesis':<14} {synthesis_model}  ~${synthesis_cost:.4f}")

    model_list = ", ".join(
        FULLOPS_WORKER_MODELS.get(k, DEFAULT_MODEL) if OPENROUTER_API_KEY else DEFAULT_MODEL
        for k in agent_keys
    )
    model_list += f", {synthesis_model} (Chief synthesis)"

    agent_names = [AGENTS[k]["name"] for k in agent_keys]
    display = (
        f"\n  Estimated cost: ~${total:.4f} — This will use {model_list}.\n"
        f"  Agents running in parallel: {', '.join(agent_names)}\n"
        + "\n".join(lines)
    )

    return display, total


def _chief_fullops_synthesis(original_prompt: str, agent_results: list[dict]) -> str:
    """
    Chief synthesizes all parallel agent outputs into one final response.
    Uses Anthropic (claude-opus-4-5) if ANTHROPIC_API_KEY is set,
    OpenRouter otherwise, Ollama as final fallback.
    """
    from core.llm_client import send_prompt

    contributions = "\n\n".join(
        f"[{r['agent']}]\n{r['response']}"
        for r in agent_results
        if r.get("is_safe", True)
    )

    synthesis_prompt = (
        f"Original request: {original_prompt}\n\n"
        f"Your team ran in parallel and produced these results:\n\n"
        f"{contributions}\n\n"
        "You are the Chief of Staff. Synthesize all of the above into one authoritative, "
        "comprehensive final response. Integrate the best insights from each agent. "
        "Do not summarize the process — deliver the answer directly to the user."
    )

    synthesis_model, backend_hint = _resolve_synthesis_backend()

    chief_config = {
        "name": "Chief",
        "model": synthesis_model,
        "system_prompt": (
            "You are the Chief of Staff — the final decision-maker. Your team has completed "
            "parallel research. You synthesize their work into the single best possible response "
            "for the user. You are authoritative, concise, and complete."
        ),
    }

    backend_label = {
        "anthropic": f"Anthropic ({synthesis_model})",
        "openrouter": f"OpenRouter ({synthesis_model})",
        "ollama": f"Ollama ({synthesis_model})",
    }.get(backend_hint, synthesis_model)

    print(f"\n  [Chief] Synthesizing via {backend_label}...", flush=True)

    try:
        return send_prompt(chief_config, synthesis_prompt, backend=backend_hint)
    except Exception as e:
        logger.error("Chief fullops synthesis failed: %s", e)
        # Fall back: return the most substantive individual result
        best = max(agent_results, key=lambda r: len(r.get("response", "")), default=None)
        return best["response"] if best else "Synthesis failed — see individual agent results."


def run_fullops(prompt: str) -> dict:
    """
    Full Ops mode: relevant agents run in parallel, Chief synthesizes the result.

    Flow:
      1. Chief determines which agents are needed
      2. Cost estimate printed — user confirms before anything runs
      3. Worker agents execute in parallel via ThreadPoolExecutor
      4. Chief synthesizes all outputs (Opus if ANTHROPIC_API_KEY, else OpenRouter/Ollama)
      5. Returns synthesized response + full individual results
    """
    from core.llm_client import send_prompt
    from core.rules_engine import build_system_prompt, scan_for_injections, wrap_untrusted_content, validate_response

    relevant_agents = _determine_relevant_agents(prompt)
    cost_display, total_cost = _estimate_fullops_cost(relevant_agents, prompt)

    print(cost_display, flush=True)
    print(f"\n  Proceed? (y/n) ", end="", flush=True)

    try:
        answer = input().strip().lower()
    except EOFError:
        answer = "n"

    if answer not in ("y", "yes"):
        print("  Aborted.", flush=True)
        return {
            "response": "Full Ops cancelled.",
            "agent": "Chief",
            "mode": "fullops",
            "elapsed_seconds": 0,
        }

    # Scan prompt for injection before distributing to agents
    scan = scan_for_injections(prompt)
    safe_prompt = wrap_untrusted_content(prompt) if scan["is_suspicious"] else prompt
    if scan["is_suspicious"]:
        logger.warning("Injection patterns in user prompt: %s", scan["threats_found"])

    def run_one_worker(agent_key: str) -> dict:
        model = FULLOPS_WORKER_MODELS.get(agent_key) if OPENROUTER_API_KEY else None
        config = _agent_config(agent_key, model_override=model)
        sys_prompt = build_system_prompt(config["name"])
        t0 = time.time()
        try:
            response = send_prompt(config, safe_prompt, system_prompt=sys_prompt)
        except Exception as e:
            logger.error("Fullops worker %s failed: %s", agent_key, e)
            return {
                "agent": config["name"],
                "response": f"Error: {e}",
                "elapsed_seconds": round(time.time() - t0, 2),
                "is_safe": False,
            }
        validation = validate_response(response)
        return {
            "agent": config["name"],
            "response": validation["sanitized_response"],
            "elapsed_seconds": round(time.time() - t0, 2),
            "is_safe": validation["is_safe"],
        }

    print(f"\n  Running {len(relevant_agents)} agents in parallel...", flush=True)
    total_start = time.time()

    worker_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(relevant_agents)) as pool:
        futures = {pool.submit(run_one_worker, key): key for key in relevant_agents}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            worker_results.append(result)
            print(f"    ✓ {result['agent']} ({result['elapsed_seconds']}s)", flush=True)

    # Chief synthesis
    final_response = _chief_fullops_synthesis(prompt, worker_results)
    total_elapsed = round(time.time() - total_start, 2)

    _, synthesis_backend = _resolve_synthesis_backend()

    return {
        "response": final_response,
        "agent": "Chief of Staff",
        "mode": "fullops",
        "synthesis_backend": synthesis_backend,
        "all_results": worker_results,
        "elapsed_seconds": total_elapsed,
    }


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def run(prompt: str, mode: str = None) -> dict:
    """
    Route a prompt through the specified (or configured) operation mode.

    Args:
        prompt: the user's input text
        mode:   "solo", "team", or "fullops" — overrides OPERATION_MODE from .env

    Returns:
        dict with: response, agent, mode, elapsed_seconds, plus mode-specific fields
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
