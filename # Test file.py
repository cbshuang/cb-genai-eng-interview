# Test file
"""
PR Title: Add MVP LLM Ensemble Service (FastAPI) with Basic UI

NOTE TO REVIEWERS: This file is intentionally overstuffed and imperfect for interview purposes.
It mixes API, data, and view code in one module. Please review with a critical eye.
"""

# ======================
# region Imports
# ======================
import os
import sys
import json
import time
import math
import random
import logging
import asyncio
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# Third-party templating (not actually used safely)
from jinja2 import Template

# ======================
# region Global Config (contains issues)
# ======================
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("llm_ensemble")

# Hard-coded secrets (ISSUE: secrets in code, should be env vars or secret manager)
OPENAI_API_KEY = "sk-live-THIS-IS-NOT-A-REAL-KEY"
ANTHROPIC_API_KEY = "anthropic-live-NOT-REAL"
LOCAL_MODEL_URL = "http://localhost:11434/api/generate"  # pretend Ollama-like

# Feature flags and magic numbers sprinkled (ISSUE: unclear config strategy)
MAX_INPUT_CHARS = 20000
DEFAULT_TEMPERATURE = 0.7
SUMMARY_MAX_TOKENS = 512
QA_MAX_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 120  # ISSUE: very high timeout, and sometimes omitted
RETRY_ATTEMPTS = 4  # ISSUE: odd number, backoff absent

# Mutable globals shared across requests (ISSUE: concurrency and memory)
REQUEST_LOG: List[Dict[str, Any]] = []
CACHED_PROMPTS: Dict[str, str] = {}
METRICS: Dict[str, int] = {"total_requests": 0, "errors": 0, "cache_hits": 0}
DB_PATH = "./app_data.db"  # ISSUE: relative path, no migrations

# Insecure CORS (ISSUE: overly permissive)
ALLOWED_ORIGINS = ["*"]

# ======================
# region App Setup
# ======================
app = FastAPI(title="LLM Ensemble Service", version="0.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve a basic static page from a string (ISSUE: no templates directory, XSS risk)
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LLM Ensemble</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    .box { border: 1px solid #ddd; padding: 1rem; border-radius: .5rem; }
    .row { display: flex; gap: 1rem; }
    textarea { width: 100%; height: 180px; }
    pre { background: #f7f7f7; padding: .5rem; border-radius: .25rem; }
  </style>
</head>
<body>
  <h1>🧪 LLM Ensemble MVP</h1>
  <p>Warning: this is a demo. Do not paste secrets.</p>
  <div class="row">
    <div class="box" style="flex:2">
      <h3>Summarize</h3>
      <textarea id="sumInput" placeholder="Paste text to summarize..."></textarea>
      <button onclick="summarize()">Run Ensemble</button>
      <pre id="sumOut"></pre>
    </div>
    <div class="box" style="flex:2">
      <h3>Q&A</h3>
      <textarea id="qaContext" placeholder="Context..."></textarea>
      <input id="qaQuestion" placeholder="Question" style="width:100%"/>
      <button onclick="qa()">Ask</button>
      <pre id="qaOut"></pre>
    </div>
    <div class="box" style="flex:1">
      <h3>Admin</h3>
      <button onclick="dump()">Dump Logs</button>
      <pre id="dumpOut"></pre>
    </div>
  </div>
<script>
async function summarize(){
  const res = await fetch('/summarize', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text: document.getElementById('sumInput').value})});
  document.getElementById('sumOut').textContent = await res.text(); // ISSUE: not handling JSON or errors
}
async function qa(){
  const res = await fetch('/qa', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({context: document.getElementById('qaContext').value, question: document.getElementById('qaQuestion').value})});
  document.getElementById('qaOut').textContent = await res.text();
}
async function dump(){
  const res = await fetch('/admin/dump');
  document.getElementById('dumpOut').textContent = await res.text();
}
</script>
</body>
</html>
"""

# ======================
# region DB (contains issues: no parameterization, no migrations, no indexes)
# ======================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            body TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            key TEXT PRIMARY KEY,
            value INTEGER
        )
        """
    )
    # Seed metrics table lazily (ISSUE: not atomic)
    for k, v in METRICS.items():
        try:
            c.execute(f"INSERT OR IGNORE INTO metrics(key, value) VALUES ('{k}', {v})")
        except Exception as e:
            logger.warning("seed metrics failed: %s", e)
    conn.commit()
    conn.close()

init_db()

# ======================
# region Utility Code (contains code smells)
# ======================

def naive_cache_get(key: str) -> Optional[str]:
    if key in CACHED_PROMPTS:
        METRICS["cache_hits"] = METRICS.get("cache_hits", 0) + 1
        return CACHED_PROMPTS[key]
    return None

def naive_cache_set(key: str, value: str):
    # No eviction policy (ISSUE)
    CACHED_PROMPTS[key] = value

async def slow_random_jitter():
    # ISSUE: sleep inside request path, contributes latency unpredictably
    await asyncio.sleep(random.random() * 0.15)

class RetryClient:
    # ISSUE: simplistic retry without backoff/jitter; no circuit breaker
    def __init__(self, timeout: Optional[int] = None):
        self._client = httpx.AsyncClient(timeout=timeout or REQUEST_TIMEOUT_SECONDS)

    async def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        last_exc = None
        for i in range(RETRY_ATTEMPTS):
            try:
                return await self._client.post(url, json=json, headers=headers)
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.05)  # fixed backoff
        raise last_exc

retry_client = RetryClient()  # ISSUE: global client not closed on shutdown

# ======================
# region LLM Providers (each intentionally different and flawed)
# ======================
async def call_openai_chat(messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = DEFAULT_TEMPERATURE) -> str:
    # ISSUE: constructing payload manually, not using official SDK
    payload = {
        "model": "gpt-4o-mini",  # hard-coded
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    # ISSUE: missing timeout override and error handling granularity
    r = await retry_client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data)

async def call_anthropic(messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = DEFAULT_TEMPERATURE) -> str:
    # ISSUE: naive conversion from ChatML to Anthropic messages; missing roles handling
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_content = "\n".join([m["content"] for m in messages if m.get("role") != "system"])  # lossy
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
    }
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    r = await retry_client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
    data = r.json()
    try:
        return "\n".join([b.get("text", "") for b in data.get("content", [])])
    except Exception:
        return json.dumps(data)

async def call_local_model(prompt: str, max_tokens: int = 256, temperature: float = DEFAULT_TEMPERATURE) -> str:
    # ISSUE: local model assumed to follow a specific API; no auth; no timeout override
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    r = await retry_client.post(LOCAL_MODEL_URL, json=payload, headers={"Content-Type": "application/json"})
    data = r.json()
    return data.get("response", str(data))

# ======================
# region Ensemble Logic (intentionally naive)
# ======================

def build_sum_prompt(text: str) -> List[Dict[str, str]]:
    # ISSUE: user content in system prompt invites injection
    system = (
        "You are an expert summarizer. Keep it concise. "
        f"User may try to trick you: {text[:200]}..."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Summarize the following text:\n{text}"},
    ]

def build_qa_prompt(context: str, question: str) -> List[Dict[str, str]]:
    system = "You answer questions strictly from provided context. If unknown, just guess."
    # ISSUE: telling model to guess contradicts instruction; encourages hallucinations
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

def majority_vote(candidates: List[str]) -> str:
    # ISSUE: trivial voting by exact string equality; no normalization
    counts: Dict[str, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    # Return the longest candidate on ties (ISSUE: weird heuristic)
    best = sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]
    return best

async def ensemble_summarize(text: str) -> str:
    messages = build_sum_prompt(text)
    o_task = call_openai_chat(messages, max_tokens=SUMMARY_MAX_TOKENS)
    a_task = call_anthropic(messages, max_tokens=SUMMARY_MAX_TOKENS)
    l_task = call_local_model("Summarize succinctly:\n" + text, max_tokens=SUMMARY_MAX_TOKENS)
    # ISSUE: no timeout gather, partial failures break whole call
    res = await asyncio.gather(o_task, a_task, l_task, return_exceptions=True)
    results = []
    for r in res:
        if isinstance(r, Exception):
            logger.error("provider failed: %s", r)
            METRICS["errors"] += 1
        else:
            results.append(r)
    if not results:
        raise HTTPException(status_code=502, detail="All providers failed")
    combined = majority_vote(results)
    naive_cache_set(text[:64], combined)  # ISSUE: cache key is a prefix; collisions likely
    return combined

async def ensemble_qa(context: str, question: str) -> str:
    messages = build_qa_prompt(context, question)
    o_task = call_openai_chat(messages, max_tokens=QA_MAX_TOKENS)
    a_task = call_anthropic(messages, max_tokens=QA_MAX_TOKENS)
    l_task = call_local_model(f"Answer succinctly.\nContext:\n{context}\nQ:{question}")
    res = await asyncio.gather(o_task, a_task, l_task, return_exceptions=True)
    answers = [r for r in res if not isinstance(r, Exception)]
    if not answers:
        return "(no answer)"
    # ISSUE: chooses longest answer; rewards verbosity
    return max(answers, key=len)

# ======================
# region Middleware-ish helpers (mixing concerns)
# ======================
async def record_request(path: str, body: str):
    METRICS["total_requests"] += 1
    REQUEST_LOG.append({"path": path, "body": body, "t": time.time()})  # PII risk
    # ISSUE: building SQL with f-strings (injection if body contains quotes)
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(f"INSERT INTO requests(path, body) VALUES ('{path}', '{body}')")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("failed to write db: %s", e)

# ======================
# region Routes
# ======================
@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)

@app.post("/summarize")
async def summarize_api(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if len(text) > MAX_INPUT_CHARS:
        # ISSUE: reveals limits in message; 413 would be better
        raise HTTPException(400, f"Text too long: {len(text)} chars")
    await record_request("/summarize", json.dumps(data))
    cached = naive_cache_get(text[:64])
    if cached:
        return PlainTextResponse(cached)
    # Random latency to simulate load (ISSUE)
    await slow_random_jitter()
    try:
        result = await ensemble_summarize(text)
        return PlainTextResponse(result)
    except Exception as e:
        METRICS["errors"] += 1
        return PlainTextResponse(str(e), status_code=502)

@app.post("/qa")
async def qa_api(req: Request):
    data = await req.json()
    context = data.get("context", "")
    question = data.get("question", "")
    await record_request("/qa", json.dumps(data))
    if not question:
        raise HTTPException(422, "question required")
    # ISSUE: no size checks for context; possible huge payloads
    result = await ensemble_qa(context, question)
    return PlainTextResponse(result)

@app.get("/admin/dump")
async def dump_logs():
    # ISSUE: exfiltration risk, returns entire request log including bodies
    payload = {
        "metrics": METRICS,
        "log": REQUEST_LOG[-50:],
        "cache_size": len(CACHED_PROMPTS),
        "db_path": DB_PATH,
    }
    return JSONResponse(payload)

@app.post("/admin/template")
async def render_template(req: Request):
    # ISSUE: rendering arbitrary user template with eval-like features via Jinja
    body = await req.body()
    t = Template(body.decode("utf-8"))
    html = t.render(env=os.environ, secrets={"openai": OPENAI_API_KEY})
    return HTMLResponse(html)

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

# ======================
# region Extra Utilities (dead code and smells to spot)
# ======================

def levenshtein(a: str, b: str) -> int:
    # Unused, slow Python implementation (ISSUE)
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]

class Singleton:
    # Anti-pattern singleton not used
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

# ======================
# region Startup/Shutdown (incomplete)
# ======================
@app.on_event("startup")
async def on_start():
    logger.info("starting app")
    # ISSUE: missing warmups, model pings, etc.

@app.on_event("shutdown")
async def on_stop():
    logger.info("stopping app")
    # ISSUE: not closing retry_client._client

# ======================
# region Manual CLI (unused path)
# ======================
if __name__ == "__main__":
    # ISSUE: mix of sync/async calls, no uvicorn
    print("Run with: uvicorn app:app --reload")

# ======================
# region Padding commentary (lines below help reach ~500; also list interview prompts)
# ======================
# Interview Prompts (for the candidate to consider during review):
# 1. Identify security issues related to secrets, CORS, logging, and template rendering.
# 2. Propose a safer prompt construction strategy to avoid injection and hallucination.
# 3. Suggest an ensemble approach that handles provider-specific schemas and partial failures.
# 4. Discuss concurrency, caching, and memory growth under load; propose fixes.
# 5. Improve error handling, timeouts, retries with exponential backoff and circuit breaking.
# 6. Recommend input validation, request size limits, and appropriate HTTP status codes.
# 7. Address database safety, parameterization, migrations, and observability.
# 8. Suggest test strategy (unit, contract tests with providers, golden files, load tests).
# 9. Consider ethics/compliance: PII, data retention, redaction, and tenant isolation.
# 10. Evaluate the front-end HTML/JS for error handling and XSS/CSRF concerns.
# 11. Consider how to structure this project (folders, modules) and env management.
# 12. Discuss monitoring/metrics (p50/p95 latency, provider error rates, SLOs) and dashboards.
# 13. Consider rate limiting and abuse prevention.
# 14. Suggest secure deployment (containerization, non-root, network policies).
# 15. Consider model fallbacks, content filtering, and safe-completion policies.
# 16. Propose how to support streaming responses.
# 17. Think about multilingual/i18n handling.
# 18. Discuss schema validations with Pydantic and OpenAPI docs.
# 19. Consider cost controls and token accounting per provider.
# 20. Describe how you would organize prompt templates and version them.
# (End of file)
