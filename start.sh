#!/usr/bin/env bash

echo ""
echo " ===================================="
echo "  DevastatorAI - Agent Starter Kit"
echo " ===================================="
echo ""

# Check Node.js is installed
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed or not in PATH."
    echo "        Download it from: https://nodejs.org/"
    echo ""
    exit 1
fi

NODE_VER=$(node --version)
echo "[OK] Node.js $NODE_VER detected."

# Copy .env.example to .env if .env does not exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp ".env.example" ".env"
        echo "[OK] .env created from .env.example. Edit it before running agents."
        echo ""
        echo "     Open .env and set your model and API key, then re-run ./start.sh"
        echo ""
        exit 0
    else
        echo "[WARN] .env.example not found. Skipping .env creation."
    fi
else
    echo "[OK] .env found."
fi

echo ""
echo " DevastatorAI is ready."
echo ""
echo " Usage:"
echo "   node core/agent_runner.js --agent rachel --prompt \"Your question here\""
echo ""
echo " Agents available:"
echo "   rachel   - Research Agent"
echo "   winter   - Writing Agent"
echo "   charlie  - Coding Agent"
echo "   chief    - Orchestration Agent"
echo "   sentinel - Security Agent"
echo ""

exec "$SHELL"
