"""
sentinel.py - System & Network Scanner for DevastatorAI

Standalone security scanner. No GUI. No Tkinter. Runs as a CLI tool.

What it does:
  1. Captures a snapshot of running processes and active network connections
  2. On first run, saves this snapshot as sentinel_baseline.json
  3. On every subsequent run, compares current state to the baseline
  4. Flags new processes and connections not present at baseline time
  5. Optionally passes anomalies to the LLM for plain-English explanation

Requirements:
    pip install psutil

Usage:
    python core/sentinel.py                        # scan and compare to baseline
    python core/sentinel.py --baseline             # force-reset the baseline
    python core/sentinel.py --explain              # pass anomalies to LLM
    python core/sentinel.py --output json          # JSON output for programmatic use

Output:
    Human-readable text by default, JSON if --output json is set.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("sentinel")

_ROOT = Path(__file__).parent.parent
BASELINE_PATH = _ROOT / "sentinel_baseline.json"

# Processes that are expected to appear/disappear between scans — suppress their alerts
PROCESS_WHITELIST = {
    "sentinel.py", "python3", "python", "node", "npm",
    "bash", "sh", "zsh", "ps", "grep", "top", "htop",
}


# ---------------------------------------------------------------------------
# psutil guard
# ---------------------------------------------------------------------------

def _require_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        print("[Sentinel] Error: psutil is not installed.")
        print("  Install it with: pip install psutil")
        sys.exit(1)


# ---------------------------------------------------------------------------
# State capture
# ---------------------------------------------------------------------------

def get_current_state() -> dict:
    """
    Capture a snapshot of:
      - running processes: pid, name, exe, cmdline, username, status
      - active network connections: local addr, remote addr, status, pid
    """
    psutil = _require_psutil()

    processes = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "username", "status"]):
        try:
            info = proc.info
            processes.append({
                "pid": info["pid"],
                "name": info.get("name") or "",
                "exe": info.get("exe") or "",
                "cmdline": " ".join(info.get("cmdline") or [])[:200],
                "username": info.get("username") or "",
                "status": info.get("status") or "",
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    connections = []
    try:
        for conn in psutil.net_connections(kind="inet"):
            laddr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
            raddr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else ""
            connections.append({
                "laddr": laddr,
                "raddr": raddr,
                "status": conn.status,
                "pid": conn.pid,
            })
    except psutil.AccessDenied:
        logger.warning("Access denied reading network connections — run with elevated privileges for full results")

    return {
        "timestamp": datetime.now().isoformat(),
        "processes": processes,
        "connections": connections,
    }


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------

def save_baseline(state: dict, path: Path = BASELINE_PATH):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[Sentinel] Baseline saved to {path}")
    print(f"           {len(state['processes'])} processes, {len(state['connections'])} connections")


def load_baseline(path: Path = BASELINE_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load baseline: %s", e)
        return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_states(baseline: dict, current: dict) -> dict:
    """
    Compare two state snapshots. Returns anomalies dict:
      new_processes    — processes in current but not in baseline (by name+exe)
      dropped_processes — processes in baseline but not in current
      new_connections  — connections in current but not in baseline (by raddr+status)
    """
    def proc_key(p):
        return (p["name"].lower(), p["exe"].lower())

    def conn_key(c):
        return (c["laddr"], c["raddr"], c["status"])

    baseline_procs = {proc_key(p) for p in baseline["processes"]}
    current_procs = {proc_key(p): p for p in current["processes"]}

    baseline_conns = {conn_key(c) for c in baseline["connections"]}
    current_conns = {conn_key(c): c for c in current["connections"]}

    new_processes = []
    for key, proc in current_procs.items():
        if key not in baseline_procs:
            name = proc["name"]
            if not any(wl.lower() in name.lower() for wl in PROCESS_WHITELIST):
                new_processes.append(proc)

    dropped_processes = []
    for key in baseline_procs:
        if key not in current_procs:
            name = key[0]
            if not any(wl.lower() in name.lower() for wl in PROCESS_WHITELIST):
                dropped_processes.append({"name": key[0], "exe": key[1]})

    new_connections = []
    for key, conn in current_conns.items():
        if key not in baseline_conns and conn.get("raddr"):
            new_connections.append(conn)

    return {
        "scan_time": current["timestamp"],
        "baseline_time": baseline["timestamp"],
        "new_processes": new_processes,
        "dropped_processes": dropped_processes,
        "new_connections": new_connections,
        "has_anomalies": bool(new_processes or new_connections),
    }


# ---------------------------------------------------------------------------
# LLM explanation (optional)
# ---------------------------------------------------------------------------

def explain_anomalies(anomalies: dict) -> str:
    """
    Send anomaly data to Sentinel agent for plain-English interpretation.
    Only called when --explain flag is passed.
    """
    sys.path.insert(0, str(_ROOT))
    from core.llm_client import send_prompt

    sentinel_config = {
        "name": "Sentinel",
        "role": "Security Agent",
        "model": "${DEFAULT_MODEL}",
        "system_prompt": (
            "You are Sentinel, a security monitoring agent. "
            "Analyze the following system anomaly report and explain in plain English:\n"
            "1. What each finding means\n"
            "2. Whether it is likely benign or suspicious\n"
            "3. Recommended action (if any)\n"
            "Use severity labels: LOW / MEDIUM / HIGH / CRITICAL."
        ),
    }

    report = json.dumps(anomalies, indent=2)
    prompt = f"Analyze these system anomalies:\n\n{report}"

    try:
        return send_prompt(sentinel_config, prompt)
    except Exception as e:
        return f"[Sentinel] LLM explanation failed: {e}"


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_report(anomalies: dict) -> str:
    lines = [
        "=" * 60,
        "  SENTINEL SECURITY SCAN REPORT",
        f"  Scan time:     {anomalies['scan_time']}",
        f"  Baseline from: {anomalies['baseline_time']}",
        "=" * 60,
        "",
    ]

    if not anomalies["has_anomalies"]:
        lines.append("  No anomalies detected. System matches baseline.")
        lines.append("")
        return "\n".join(lines)

    new_procs = anomalies.get("new_processes", [])
    if new_procs:
        lines.append(f"  [!] NEW PROCESSES ({len(new_procs)} found):")
        for p in new_procs:
            lines.append(f"      PID {p['pid']:6d}  {p['name']:<30s}  {p['username']}")
            if p.get("cmdline"):
                lines.append(f"             CMD: {p['cmdline'][:80]}")
        lines.append("")

    dropped_procs = anomalies.get("dropped_processes", [])
    if dropped_procs:
        lines.append(f"  [i] DROPPED PROCESSES ({len(dropped_procs)} found):")
        for p in dropped_procs:
            lines.append(f"      {p['name']} ({p['exe'] or 'unknown path'})")
        lines.append("")

    new_conns = anomalies.get("new_connections", [])
    if new_conns:
        lines.append(f"  [!] NEW NETWORK CONNECTIONS ({len(new_conns)} found):")
        for c in new_conns:
            lines.append(
                f"      {c['laddr']:<25s} -> {c['raddr']:<25s}  [{c['status']}]  pid={c['pid']}"
            )
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------

def scan(force_baseline: bool = False, explain: bool = False, output_format: str = "text") -> dict:
    """
    Run a full Sentinel scan.

    Args:
        force_baseline: if True, save current state as new baseline and exit
        explain:        if True, pass anomalies to LLM for plain-English analysis
        output_format:  "text" or "json"

    Returns:
        dict — anomalies report (or baseline info if force_baseline=True)
    """
    current = get_current_state()

    if force_baseline:
        save_baseline(current)
        return {"status": "baseline_saved", "timestamp": current["timestamp"]}

    baseline = load_baseline()

    if baseline is None:
        print("[Sentinel] No baseline found. Saving current state as baseline.")
        save_baseline(current)
        return {"status": "baseline_created", "timestamp": current["timestamp"]}

    anomalies = compare_states(baseline, current)

    if output_format == "json":
        return anomalies

    # Text output
    print(format_report(anomalies))

    if explain and anomalies["has_anomalies"]:
        print("\n[Sentinel] Sending anomalies to LLM for analysis...\n")
        explanation = explain_anomalies(anomalies)
        print(explanation)
        anomalies["llm_explanation"] = explanation

    return anomalies


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Sentinel — DevastatorAI Security Scanner")
    parser.add_argument("--baseline", action="store_true",
                        help="Reset baseline to current system state")
    parser.add_argument("--explain", action="store_true",
                        help="Pass anomalies to LLM for plain-English explanation")
    parser.add_argument("--output", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    args = parser.parse_args()

    result = scan(
        force_baseline=args.baseline,
        explain=args.explain,
        output_format=args.output,
    )

    if args.output == "json":
        print(json.dumps(result, indent=2))
