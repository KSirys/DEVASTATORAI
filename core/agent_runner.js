#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || '';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'qwen2.5:7b';

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--agent' && argv[i + 1]) args.agent = argv[++i];
    else if (argv[i] === '--prompt' && argv[i + 1]) args.prompt = argv[++i];
    else if (argv[i] === '--model' && argv[i + 1]) args.model = argv[++i];
  }
  return args;
}

function loadAgent(agentName) {
  const agentsDir = path.join(__dirname, '..', 'agents');
  const files = fs.readdirSync(agentsDir).filter(f => f.endsWith('.json'));

  for (const file of files) {
    const config = JSON.parse(fs.readFileSync(path.join(agentsDir, file), 'utf8'));
    if (config.name.toLowerCase() === agentName.toLowerCase() ||
        file.replace('.json', '').toLowerCase() === agentName.toLowerCase()) {
      return config;
    }
  }
  throw new Error(`Agent "${agentName}" not found in agents/ folder.`);
}

function resolveModel(agentModel, cliModel) {
  if (cliModel) return cliModel;
  if (agentModel && agentModel !== '${DEFAULT_MODEL}') return agentModel;
  return DEFAULT_MODEL;
}

function postJSON(url, data, headers) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(data);
    const parsed = new URL(url);
    const options = {
      hostname: parsed.hostname,
      port: parsed.port,
      path: parsed.pathname,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body), ...headers }
    };
    const lib = parsed.protocol === 'https:' ? https : http;
    const req = lib.request(options, res => {
      let raw = '';
      res.on('data', chunk => { raw += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body: raw }));
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

async function runOllama(model, systemPrompt, userPrompt) {
  const url = `${OLLAMA_URL}/api/chat`;
  const payload = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ],
    stream: false
  };
  const result = await postJSON(url, payload, {});
  if (result.status !== 200) throw new Error(`Ollama error ${result.status}: ${result.body}`);
  const data = JSON.parse(result.body);
  return data.message?.content || data.response || '';
}

async function runOpenRouter(model, systemPrompt, userPrompt) {
  const url = 'https://openrouter.ai/api/v1/chat/completions';
  const payload = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ]
  };
  const headers = {
    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
    'HTTP-Referer': 'https://github.com/KSirys/DevastatorAI',
    'X-Title': 'DevastatorAI'
  };
  const result = await postJSON(url, payload, headers);
  if (result.status !== 200) throw new Error(`OpenRouter error ${result.status}: ${result.body}`);
  const data = JSON.parse(result.body);
  return data.choices?.[0]?.message?.content || '';
}

async function main() {
  const args = parseArgs(process.argv);

  if (!args.agent) {
    console.error('Usage: node core/agent_runner.js --agent <name> --prompt "<text>" [--model <model>]');
    console.error('');
    console.error('Available agents:');
    const agentsDir = path.join(__dirname, '..', 'agents');
    fs.readdirSync(agentsDir).filter(f => f.endsWith('.json')).forEach(f => {
      const cfg = JSON.parse(fs.readFileSync(path.join(agentsDir, f), 'utf8'));
      console.error(`  ${cfg.name} — ${cfg.role}`);
    });
    process.exit(1);
  }

  if (!args.prompt) {
    console.error('Error: --prompt is required.');
    process.exit(1);
  }

  const agent = loadAgent(args.agent);
  const model = resolveModel(agent.model, args.model);
  const systemPrompt = agent.system_prompt || `You are ${agent.name}, a ${agent.role}.`;

  console.error(`[DevastatorAI] Agent: ${agent.name} | Model: ${model} | Backend: ${OPENROUTER_API_KEY ? 'OpenRouter' : 'Ollama'}`);

  let response;
  if (OPENROUTER_API_KEY) {
    response = await runOpenRouter(model, systemPrompt, args.prompt);
  } else {
    response = await runOllama(model, systemPrompt, args.prompt);
  }

  console.log(response);
}

main().catch(err => {
  console.error('[DevastatorAI] Error:', err.message);
  process.exit(1);
});
