<p align="center">
  <img src="assets/logo.png" alt="DevastatorAI" width="400"/>
</p>

# DevastatorAI

**Open source multi-agent AI starter kit. Build your AI team, your way.**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Node](https://img.shields.io/badge/Node-20%2B-brightgreen.svg)
![Ollama](https://img.shields.io/badge/Ollama-supported-orange.svg)
![OpenRouter](https://img.shields.io/badge/OpenRouter-supported-purple.svg)

---

## What is DevastatorAI?

DevastatorAI is a multi-agent AI framework that ships with five pre-configured agents, three operation modes, and a built-in security rules engine. It runs entirely on your machine using Ollama, or connects to cloud models via OpenRouter — no infrastructure required. Every prompt is screened before it reaches an agent, and every response is validated before it reaches you. DevastatorAI is the engine that powers the [Devastator Dashboard](https://github.com/KSirys/DEVASTATOR-dashboard).

---

## The Five Agents

| Agent | Role | What It Does |
|---|---|---|
| **Rachel** | Research Agent | Searches and summarizes information from the web or your own context |
| **Winter** | Writing Agent | Drafts content, documents, emails, and long-form copy |
| **Charlie** | Coding Agent | Writes, reviews, and debugs code across any language |
| **Chief of Staff** | Orchestration Agent | Classifies your input and routes it to the right agent automatically |
| **Sentinel** | Security Agent | Monitors running processes and network connections for anomalies |

---

## Three Operation Modes

Set `OPERATION_MODE` in your `.env` to switch modes at any time.

| Mode | Cost | How It Works | Best For |
|---|---|---|---|
| **solo** | $0 | One agent handles the full task. Local Ollama only. No API key needed. | Personal use on any machine |
| **team** | Low (~$0.01–0.10/session) | Chief classifies your input and routes to 2–3 agents sequentially. Uses OpenRouter free tier if a key is set, otherwise Ollama. | Collaborative tasks that benefit from multiple perspectives |
| **fullops** | Medium (~$0.50–5.00/session) | All relevant agents run in parallel using the best available models. Cost is estimated and confirmed before the run starts. | Production workloads where quality matters more than cost |

---

## Quick Start

```bash
git clone https://github.com/KSirys/DEVASTATORAI.git
cd DEVASTATORAI
```

**Windows:**
```
start.bat
```

**Linux / Mac:**
```bash
chmod +x start.sh
./start.sh
```

On first run, `.env` is created from `.env.example`. Open it, set your Ollama URL or OpenRouter API key, then run again.

**Send a prompt:**
```bash
node index.js "Research the top open source LLM frameworks"
node index.js "Write a blog post about AI agents" --mode team
node index.js "Review this architecture for security issues" --mode fullops
```

**Call a specific agent directly:**
```bash
python core/agent_runner.py --agent charlie --prompt "Write a binary search in Python"
python core/agent_runner.py --list
```

---

## Configuration

All settings live in `.env`. Copy `.env.example` to get started.

| Variable | Default | Description |
|---|---|---|
| `OPERATION_MODE` | `solo` | Operation mode: `solo`, `team`, or `fullops` |
| `OLLAMA_URL` | `http://localhost:11434` | URL of your local Ollama instance |
| `OPENROUTER_API_KEY` | _(blank)_ | OpenRouter API key. Leave blank to use Ollama only. |
| `DEFAULT_MODEL` | `qwen2.5:7b` | Model to use. Ollama model name or OpenRouter model path. |
| `SEARCH_PROVIDER` | `duckduckgo` | Web search provider: `duckduckgo`, `google`, or `brave` |
| `ANTHROPIC_API_KEY` | _(blank)_ | Optional. Enables Claude Opus as Chief synthesis model in Full Ops mode. |

---

## Operation Mode Details

### Solo
One agent handles the full task from start to finish. No API key required — runs entirely on local Ollama. Chief classifies your prompt using keyword matching and LLM fallback, selects the most appropriate agent, and returns the response. The fastest and cheapest option.

```
You → Chief (classifies) → Rachel/Winter/Charlie/Sentinel → You
```

### Team
Chief builds a sequential pipeline of 2–3 agents tailored to the task. The pipeline is printed before execution begins. Each agent's output becomes context for the next. At the end, Chief reviews all contributions and returns a polished final response.

**Example — research task:**
```
You → Chief → Rachel (research) → Winter (write summary) → Chief review → You
```

**Example — coding task:**
```
You → Chief → Charlie (write code) → Rachel (accuracy review) → Chief review → You
```

Uses OpenRouter free tier models (`nvidia/llama-3.1-nemotron-70b-instruct`) if `OPENROUTER_API_KEY` is set. Falls back to Ollama otherwise.

### Full Ops
Chief determines the most relevant agents for the task, runs them in **parallel** using the best available models, then synthesizes all outputs into one authoritative final response.

Before anything runs, the cost is estimated and printed:
```
Estimated cost: ~$0.0023 — This will use nvidia/llama-3.1-nemotron-70b-instruct (Rachel, Winter),
deepseek/deepseek-r1 (Charlie), claude-opus-4-5 (Chief synthesis).
Proceed? (y/n)
```

**Worker models (OpenRouter):**
- Rachel, Winter: `nvidia/llama-3.1-nemotron-70b-instruct` (free tier)
- Charlie: `deepseek/deepseek-r1`
- Sentinel: `nvidia/llama-3.1-nemotron-70b-instruct`

**Chief synthesis model:**
- If `ANTHROPIC_API_KEY` is set: `claude-opus-4-5` (Anthropic direct)
- If only `OPENROUTER_API_KEY` is set: `nvidia/llama-3.1-nemotron-70b-instruct`
- If neither: `DEFAULT_MODEL` via Ollama

---

## Search Providers

Rachel and other research tasks use web search when needed. Configure via `SEARCH_PROVIDER` in `.env`.

| Provider | Cost | Keys Required | Notes |
|---|---|---|---|
| **DuckDuckGo** | Free | None | Default. Works out of the box. |
| **Google** | Paid | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | Most accurate. [Get a key](https://developers.google.com/custom-search/v1/overview) |
| **Brave** | Free tier | `BRAVE_API_KEY` | 2,000 queries/month free. [Get a key](https://api.search.brave.com/) |

---

## Security

Every interaction passes through a three-layer rules engine before anything reaches the LLM or the user.

**Layer 1 — Universal Rules** (`core/rules_engine.py`)
Injected as a system prompt on every request to every agent. Covers: trust boundaries, absolute prohibitions, prompt injection defense, output sanitization, and fail-safe behavior.

**Layer 2 — Per-Agent Rules**
Additional rules loaded from `configs/rules.json` for specific agents. Customize what each agent is and isn't allowed to do.

**Layer 3 — Response Validation**
Every response is scanned before it reaches you. Catches credential leaks (API keys, tokens, private keys), injection echoes, and blocklist matches. Blocked responses are replaced with a clear explanation of what was caught.

Rules are fully configurable via `configs/rules.json`. Built-in defaults are used automatically when the file is absent.

---

## Sentinel Agent

Sentinel is a standalone security scanner that monitors your system for changes between runs.

**How it works:**
1. First run saves a baseline snapshot of all running processes and active network connections
2. Every subsequent run compares the current state to the baseline
3. New processes and connections not present at baseline are flagged as anomalies
4. Pass `--explain` to send anomalies to the LLM for plain-English analysis

**Run it:**
```bash
# Scan and compare to baseline
python core/sentinel.py

# Reset baseline to current state
python core/sentinel.py --baseline

# Scan + LLM explanation of anomalies
python core/sentinel.py --explain

# JSON output for programmatic use
python core/sentinel.py --output json
```

Requires: `pip install psutil`

---

## Powered By

| Tool | Purpose |
|---|---|
| [Ollama](https://ollama.com/) | Local LLM inference — run any open model on your hardware |
| [OpenRouter](https://openrouter.ai/) | Cloud LLM gateway — access GPT-4, Claude, Gemini, and more |
| [psutil](https://github.com/giampaolo/psutil) | System process and network monitoring for Sentinel |
| [DuckDuckGo](https://duckduckgo.com/) | Free web search, no API key required |

---

## Related

**Devastator Dashboard** — the full AI command center built on DevastatorAI, featuring an alien bridge cockpit UI, widget docking, and live agent panels.

[github.com/KSirys/DEVASTATOR-dashboard](https://github.com/KSirys/DEVASTATOR-dashboard)

---

## License

MIT License — Copyright © 2026 KSirys

See [LICENSE](LICENSE) for full terms.
