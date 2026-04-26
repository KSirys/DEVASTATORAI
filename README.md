<p align="center">
  <img src="assets/logo.png" alt="DevastatorAI" width="400"/>
</p>

# DevastatorAI

Open source multi-agent AI starter kit. Ships with 5 pre-configured agents and works with Ollama locally or OpenRouter via API key. It is the engine that powers the Devastator Dashboard.

---

## What It Is

DevastatorAI is a lightweight, modular AI agent framework for running local or cloud-hosted language models. You define agents as simple JSON configs, point the runner at Ollama or OpenRouter, and send prompts from the command line or any application that calls it.

No heavy frameworks. No lock-in. Just agents, a runner, and your models.

---

## Agents Included

| Agent | Role | Description |
|---|---|---|
| Rachel | Research Agent | Searches and summarizes information |
| Winter | Writing Agent | Drafts content and documents |
| Charlie | Coding Agent | Writes and reviews code |
| Chief of Staff | Orchestration Agent | Routes tasks to the right agent |
| Sentinel | Security Agent | Monitors system processes and network activity for anomalies |

---

## Requirements

- [Node.js](https://nodejs.org/) v18 or higher
- [Ollama](https://ollama.com/) (for local models) **or** an [OpenRouter](https://openrouter.ai/) API key (for cloud models)

---

## Quick Start

```bash
git clone https://github.com/KSirys/DevastatorAI.git
cd DevastatorAI
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

On first run, `.env` is created from `.env.example`. Edit it to set your model and API key.

To send a prompt:
```bash
node core/agent_runner.js --agent rachel --prompt "Summarize the history of neural networks"
```

---

## Connect Ollama

1. Install Ollama from [https://ollama.com/](https://ollama.com/)
2. Pull a model: `ollama pull qwen2.5:7b`
3. In `.env`, set:
   ```
   OLLAMA_URL=http://localhost:11434
   DEFAULT_MODEL=qwen2.5:7b
   OPENROUTER_API_KEY=
   ```
4. Leave `OPENROUTER_API_KEY` blank to use Ollama automatically.

---

## Connect OpenRouter

1. Get an API key from [https://openrouter.ai/](https://openrouter.ai/)
2. In `.env`, set:
   ```
   OPENROUTER_API_KEY=your_key_here
   DEFAULT_MODEL=openai/gpt-4o
   ```
3. When `OPENROUTER_API_KEY` is set, the runner uses OpenRouter instead of Ollama.

---

## Devastator Dashboard

DevastatorAI is the backend engine for the Devastator Dashboard — a local AI command center with an alien bridge cockpit UI.

[github.com/KSirys/DEVASTATOR-dashboard](https://github.com/KSirys/DEVASTATOR-dashboard)

---

## License

MIT — see [LICENSE](LICENSE)
