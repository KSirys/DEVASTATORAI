"""
web_search.py - Multi-Provider Web Search for DevastatorAI

Configurable via SEARCH_PROVIDER in .env. Supported providers:

  duckduckgo  (default) — free, no key required
  google                — requires GOOGLE_API_KEY + GOOGLE_CSE_ID in .env
  brave                 — requires BRAVE_API_KEY in .env

Configuration examples:

  # DuckDuckGo (default, works out of the box)
  SEARCH_PROVIDER=duckduckgo

  # Google Custom Search
  SEARCH_PROVIDER=google
  GOOGLE_API_KEY=your_google_api_key_here
  GOOGLE_CSE_ID=your_custom_search_engine_id_here
  # Get API key: https://developers.google.com/custom-search/v1/overview
  # Set up CSE:  https://programmablesearchengine.google.com/

  # Brave Search
  SEARCH_PROVIDER=brave
  BRAVE_API_KEY=your_brave_api_key_here
  # Get API key: https://api.search.brave.com/

Public API:
    search(query, max_results=5)        -> dict
    format_results_for_agent(result)    -> str
"""

import os
import re
import json
import urllib.request
import urllib.parse
import urllib.error
import logging
from pathlib import Path

logger = logging.getLogger("web_search")


# ---------------------------------------------------------------------------
# Load env config
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

SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER") or _env.get("SEARCH_PROVIDER", "duckduckgo")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or _env.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID") or _env.get("GOOGLE_CSE_ID", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY") or _env.get("BRAVE_API_KEY", "")


# ---------------------------------------------------------------------------
# Result format
# ---------------------------------------------------------------------------

def _result(success, query, results, error=None):
    return {"success": success, "query": query, "results": results, "error": error}


# ---------------------------------------------------------------------------
# Provider: DuckDuckGo (free, no key)
# ---------------------------------------------------------------------------

def _clean_html(text):
    clean = re.sub(r'<[^>]+>', '', text)
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                          ("&quot;", '"'), ("&#x27;", "'"), ("&nbsp;", " ")]:
        clean = clean.replace(entity, char)
    return re.sub(r'\s+', ' ', clean).strip()


def _ddg_parse_lite(html, max_results):
    results = []

    link_pat = re.compile(r'<a[^>]*class="result-link"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
    snip_pat = re.compile(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', re.DOTALL)

    links = link_pat.findall(html)
    snippets = snip_pat.findall(html)

    for i in range(min(len(links), len(snippets), max_results)):
        url = links[i][0].strip()
        title = _clean_html(links[i][1])
        snippet = _clean_html(snippets[i])
        if url and title:
            results.append({"title": title, "snippet": snippet, "url": url})

    # Fallback: generic link extraction
    if not results:
        for url, title_html in re.compile(
            r'<a[^>]*href="(https?://[^"]*)"[^>]*>(.*?)</a>', re.DOTALL
        ).findall(html)[:max_results]:
            if "duckduckgo.com" in url:
                continue
            title = _clean_html(title_html)
            if title and len(title) > 5:
                results.append({"title": title, "snippet": "", "url": url})

    return results


def _ddg_instant_answer(query):
    """DuckDuckGo instant answer API — fallback when lite page yields nothing."""
    try:
        encoded = urllib.parse.quote_plus(query)
        req = urllib.request.Request(
            f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1",
            headers={"User-Agent": "DevastatorAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", "Answer"),
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
            })
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic["Text"][:80],
                    "snippet": topic["Text"],
                    "url": topic.get("FirstURL", ""),
                })
        return results
    except Exception as e:
        logger.error("DDG instant answer failed: %s", e)
        return []


def search_duckduckgo(query, max_results=5):
    """Search DuckDuckGo lite. Falls back to instant answer API if HTML parsing yields nothing."""
    try:
        encoded = urllib.parse.quote_plus(query)
        req = urllib.request.Request(
            f"https://lite.duckduckgo.com/lite/?q={encoded}",
            headers={"User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        results = _ddg_parse_lite(html, max_results)
        if not results:
            results = _ddg_instant_answer(query)

        return _result(True, query, results)
    except urllib.error.URLError as e:
        logger.error("DuckDuckGo search failed: %s", e)
        return _result(False, query, [], f"Network error: {e}")
    except Exception as e:
        logger.error("DuckDuckGo search error: %s", e)
        return _result(False, query, [], str(e))


# ---------------------------------------------------------------------------
# Provider: Google Custom Search API
# Set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env
# ---------------------------------------------------------------------------

def search_google(query, max_results=5):
    """
    Google Custom Search JSON API.
    Requires: GOOGLE_API_KEY and GOOGLE_CSE_ID in .env
    Get API key: https://developers.google.com/custom-search/v1/overview
    Set up CSE:  https://programmablesearchengine.google.com/
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return _result(False, query, [],
            "Google search requires GOOGLE_API_KEY and GOOGLE_CSE_ID in .env")

    try:
        params = urllib.parse.urlencode({
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": min(max_results, 10),
        })
        req = urllib.request.Request(
            f"https://www.googleapis.com/customsearch/v1?{params}",
            headers={"User-Agent": "DevastatorAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        results = []
        for item in data.get("items", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
            })
        return _result(True, query, results)
    except urllib.error.HTTPError as e:
        logger.error("Google search HTTP error: %s", e)
        return _result(False, query, [], f"Google API error {e.code}: {e.reason}")
    except Exception as e:
        logger.error("Google search error: %s", e)
        return _result(False, query, [], str(e))


# ---------------------------------------------------------------------------
# Provider: Brave Search API
# Set BRAVE_API_KEY in .env
# ---------------------------------------------------------------------------

def search_brave(query, max_results=5):
    """
    Brave Search API.
    Requires: BRAVE_API_KEY in .env
    Get API key: https://api.search.brave.com/
    Free tier: 2,000 queries/month
    """
    if not BRAVE_API_KEY:
        return _result(False, query, [],
            "Brave search requires BRAVE_API_KEY in .env")

    try:
        params = urllib.parse.urlencode({"q": query, "count": min(max_results, 20)})
        req = urllib.request.Request(
            f"https://api.search.brave.com/res/v1/web/search?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("description", ""),
                "url": item.get("url", ""),
            })
        return _result(True, query, results)
    except urllib.error.HTTPError as e:
        logger.error("Brave search HTTP error: %s", e)
        return _result(False, query, [], f"Brave API error {e.code}: {e.reason}")
    except Exception as e:
        logger.error("Brave search error: %s", e)
        return _result(False, query, [], str(e))


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def search(query, max_results=5):
    """
    Run a web search using the provider set in SEARCH_PROVIDER (.env).
    Falls back to DuckDuckGo if the configured provider fails.

    Returns:
        dict: {success, query, results: [{title, snippet, url}], error}
    """
    provider = SEARCH_PROVIDER.lower().strip()

    if provider == "google":
        logger.info("Searching via Google: %s", query)
        return search_google(query, max_results)
    elif provider == "brave":
        logger.info("Searching via Brave: %s", query)
        return search_brave(query, max_results)
    else:
        # duckduckgo or any unknown value
        if provider not in ("duckduckgo", "ddg", ""):
            logger.warning("Unknown SEARCH_PROVIDER '%s', falling back to DuckDuckGo", provider)
        logger.info("Searching via DuckDuckGo: %s", query)
        return search_duckduckgo(query, max_results)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def format_results_for_agent(search_data):
    """
    Convert search results dict into a plain-text block for agent context.
    This output is then passed through wrap_untrusted_content() in the caller.
    """
    if not search_data["success"]:
        return f"Search failed: {search_data['error']}"
    if not search_data["results"]:
        return f"No results found for: {search_data['query']}"

    lines = [f"Web search results for: {search_data['query']}\n"]
    for i, r in enumerate(search_data["results"], 1):
        lines.append(f"[{i}] {r['title']}")
        if r.get("snippet"):
            lines.append(f"    {r['snippet']}")
        if r.get("url"):
            lines.append(f"    Source: {r['url']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== Web Search Test ===")
    print(f"Provider: {SEARCH_PROVIDER}\n")

    result = search("open source AI agent frameworks", max_results=3)

    if result["success"]:
        print(f"Found {len(result['results'])} results:\n")
        print(format_results_for_agent(result))
    else:
        print(f"Search failed: {result['error']}")

    print("=== Done ===")
