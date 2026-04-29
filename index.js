#!/usr/bin/env node
/**
 * index.js - DevastatorAI Main Entry Point
 *
 * Loads .env, displays active operation mode, accepts a prompt from the
 * command line, passes it to core/chief.py, and prints the response with
 * the agent name that handled it.
 *
 * Usage:
 *   node index.js "Your prompt here"
 *   node index.js "Your prompt here" --mode team
 *   node index.js "Your prompt here" --mode fullops
 *   node index.js --mode solo          (interactive: prompts for input)
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const ROOT = __dirname;

// ---------------------------------------------------------------------------
// Load .env
// ---------------------------------------------------------------------------

function loadEnv() {
  const envPath = path.join(ROOT, '.env');
  const examplePath = path.join(ROOT, '.env.example');
  const target = fs.existsSync(envPath) ? envPath : (fs.existsSync(examplePath) ? examplePath : null);

  if (!target) return {};

  const env = {};
  fs.readFileSync(target, 'utf8').split('\n').forEach(line => {
    line = line.trim();
    if (!line || line.startsWith('#') || !line.includes('=')) return;
    const [key, ...rest] = line.split('=');
    env[key.trim()] = rest.join('=').trim();
  });
  return env;
}

const dotenv = loadEnv();
const OPERATION_MODE = process.env.OPERATION_MODE || dotenv.OPERATION_MODE || 'solo';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || dotenv.OPENROUTER_API_KEY || '';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || dotenv.DEFAULT_MODEL || 'qwen2.5:7b';

// ---------------------------------------------------------------------------
// Display banner
// ---------------------------------------------------------------------------

function printBanner() {
  const backend = OPENROUTER_API_KEY ? 'OpenRouter' : 'Ollama (local)';
  console.log('');
  console.log('  ╔══════════════════════════════════╗');
  console.log('  ║       D E V A S T A T O R A I    ║');
  console.log('  ╚══════════════════════════════════╝');
  console.log(`  Mode:    ${OPERATION_MODE.toUpperCase()}`);
  console.log(`  Backend: ${backend}`);
  console.log(`  Model:   ${DEFAULT_MODEL}`);
  console.log('');
}

// ---------------------------------------------------------------------------
// Detect Python binary
// ---------------------------------------------------------------------------

function findPython() {
  for (const bin of ['python3', 'python']) {
    const result = spawnSync(bin, ['--version'], { encoding: 'utf8' });
    if (result.status === 0) return bin;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Call chief.py
// ---------------------------------------------------------------------------

function callChief(prompt, mode) {
  const python = findPython();
  if (!python) {
    console.error('[Error] Python 3 is not installed or not in PATH.');
    process.exit(1);
  }

  const chiefPath = path.join(ROOT, 'core', 'chief.py');
  if (!fs.existsSync(chiefPath)) {
    console.error(`[Error] core/chief.py not found at ${chiefPath}`);
    process.exit(1);
  }

  const args = ['core/chief.py', '--prompt', prompt];
  if (mode) args.push('--mode', mode);

  const result = spawnSync(python, args, {
    cwd: ROOT,
    encoding: 'utf8',
    stdio: ['inherit', 'pipe', 'pipe'],
    timeout: 600_000, // 10-minute timeout
  });

  if (result.error) {
    console.error(`[Error] Failed to launch chief.py: ${result.error.message}`);
    process.exit(1);
  }

  if (result.stderr) {
    // Print Python warnings/logs at low volume (filter noise)
    const stderrLines = result.stderr.split('\n').filter(l =>
      l.trim() && !l.includes('DeprecationWarning') && !l.includes('ResourceWarning')
    );
    if (stderrLines.length) {
      stderrLines.forEach(l => process.stderr.write(`  [py] ${l}\n`));
    }
  }

  if (result.status !== 0) {
    console.error(`[Error] chief.py exited with code ${result.status}`);
    if (result.stdout) console.error(result.stdout);
    process.exit(result.status);
  }

  // chief.py outputs a single JSON line
  try {
    return JSON.parse(result.stdout.trim());
  } catch {
    // If not JSON, return raw text (e.g. fullops cost confirmation flow)
    return { response: result.stdout.trim(), agent: 'Chief', mode };
  }
}

// ---------------------------------------------------------------------------
// Interactive mode (no prompt given on CLI)
// ---------------------------------------------------------------------------

async function interactiveMode(mode) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const ask = (q) => new Promise(resolve => rl.question(q, resolve));

  console.log('  Type your prompt and press Enter. Ctrl+C to exit.\n');

  while (true) {
    const prompt = await ask('  You: ');
    if (!prompt.trim()) continue;

    console.log('');
    const data = callChief(prompt.trim(), mode);
    printResponse(data);
    console.log('');
  }
}

// ---------------------------------------------------------------------------
// Format and print response
// ---------------------------------------------------------------------------

function printResponse(data) {
  const agent = data.agent || 'Unknown';
  const elapsed = data.elapsed_seconds != null ? ` (${data.elapsed_seconds}s)` : '';
  const mode = data.mode ? ` [${data.mode}]` : '';

  console.log(`  ┌─ ${agent}${mode}${elapsed}`);
  console.log('  │');

  const lines = (data.response || '').split('\n');
  lines.forEach(line => console.log(`  │  ${line}`));

  console.log('  └─────────────────────────────────');

  // In team/fullops mode, optionally show all agent contributions
  if (data.team_results && data.team_results.length > 1) {
    console.log('\n  [Team contributions]');
    data.team_results.forEach(r => {
      if (r.agent !== agent) {
        console.log(`  ├─ ${r.agent} (${r.elapsed_seconds}s): ${r.response.slice(0, 120)}...`);
      }
    });
  }
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  let prompt = null;
  let mode = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--mode' && args[i + 1]) {
      mode = args[++i];
    } else if (!args[i].startsWith('--')) {
      prompt = args[i];
    }
  }

  return { prompt, mode };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

printBanner();

const { prompt, mode } = parseArgs();
const activeMode = mode || OPERATION_MODE;

if (prompt) {
  // Single-shot mode: prompt given on command line
  console.log(`  Prompt: "${prompt}"\n`);
  const data = callChief(prompt, activeMode);
  printResponse(data);
  console.log('');
} else {
  // Interactive mode: read prompts from stdin
  interactiveMode(activeMode).catch(err => {
    if (err.code !== 'ERR_USE_AFTER_CLOSE') {
      console.error(err);
      process.exit(1);
    }
  });
}
