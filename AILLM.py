"""
PR Title: Add Minimal LLM Duo Service (FastAPI)

Purpose
-------
A very small service that calls two LLM providers and fuses their answers.
Deliberately includes **3–5 open questions** (left as TODOs) for exploration in the review.
"""

from __future__ import annotations
import os
import json
import asyncio
from typing import Dict, Any, List, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# Configuration (kept compact on purpose)
# -----------------------------------------------------------------------------
# NOTE: These default to placeholders so the app can boot in demo environments.
# Open question: Should we fail-fast if keys are missing, or allow degraded/no-op?
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "changeme-openai")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "changeme-anthropic")
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "15"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

app = FastAPI(title="Minimal LLM Duo", version="0.1.0")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class Providers:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)

    async def openai(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        r = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def anthropic(self, messages: List[Dict[str, str]]) -> str:
        # Simple mapping to Anthropic 'messages' format.
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_concat = "\n".join(m["content"] for m in messages if m.get("role") != "system")
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "system": system,
            "messages": [{"role": "user", "content": user_concat}],
        }
        r = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        data = r.json()
        parts = data.get("content", [])
        return "\n".join(p.get("text", "") for p in parts)

    async def close(self) -> None:
        await self.client.aclose()

providers = Providers()

# -----------------------------------------------------------------------------
# Prompting
# -----------------------------------------------------------------------------
# Open question 1 (Prompt safety):
#   The endpoint allows optional "instructions" that are appended to the system prompt
#   as-is. What guardrails (sanitization, allow-listing, segmentation) would you add
#   to minimize prompt injection / jailbreaking while preserving flexibility?

def build_messages(task: str, content: str, instructions: str | None = None) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a helpful assistant. Be concise when possible."
        + (f" Instructions: {instructions}" if instructions else "")
    )
    # Open question 2 (Attribution & grounding):
    #   We encourage citing sources when the user provides URLs in content, but we do not
    #   currently parse/validate those. How would you design a grounding layer and
    #   evidence schema (e.g., citations with confidence) without overcomplicating the API?
    user_prompt = f"Task: {task}\nContent:\n{content}"
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------
# Open question 3 (Fusion policy):
#   We return the *shorter* answer unless one model explicitly states "I don't know",
#   in which case we prefer the other. What alternative fusion/consensus strategies
#   (e.g., semantic agreement, majority vote on extracted claims, RAG cross-checking)
#   would you propose for reliability and cost?

def fuse(a: str, b: str) -> Tuple[str, Dict[str, Any]]:
    a_unknown = "i don't know" in a.lower() or "cannot answer" in a.lower()
    b_unknown = "i don't know" in b.lower() or "cannot answer" in b.lower()
    if a_unknown and not b_unknown:
        winner = b
    elif b_unknown and not a_unknown:
        winner = a
    else:
        winner = a if len(a) <= len(b) else b
    meta = {"a_len": len(a), "b_len": len(b), "chose": "a" if winner is a else "b"}
    return winner, meta

# -----------------------------------------------------------------------------
# In-memory cache
# -----------------------------------------------------------------------------
# Open question 4 (Caching & privacy):
#   We cache by (task, hash(content)) for 10 minutes in-memory.
#   Is in-memory sufficient for multi-instance deployments? How should TTLs and
#   PII/tenant isolation be handled? Would you add content hashing, k-Anonymity, or
#   disable caching for certain tasks?

import time
_cache: Dict[str, Tuple[float, str]] = {}
CACHE_TTL_S = 600

def cache_get(key: str) -> str | None:
    rec = _cache.get(key)
    if not rec:
        return None
    ts, val = rec
    if time.time() - ts > CACHE_TTL_S:
        _cache.pop(key, None)
        return None
    return val

def cache_set(key: str, val: str) -> None:
    _cache[key] = (time.time(), val)

# -----------------------------------------------------------------------------
# Endpoint
# -----------------------------------------------------------------------------
# POST /analyze
# body: { task: "summarize" | "qa", content: string, instructions?: string }
# returns: { answer: string, meta: {...} }
#
# Open question 5 (Resilience & observability):
#   We gather results with a fixed timeout and no retries. What is the right retry
#   policy/backoff? How would you instrument this endpoint (latency, error budget,
#   provider histograms) and set SLOs?

@app.post("/analyze")
async def analyze(body: Dict[str, Any]):
    task = (body.get("task") or "").strip().lower()
    content = body.get("content") or ""
    instructions = body.get("instructions")

    if task not in {"summarize", "qa"}:
        raise HTTPException(status_code=422, detail="task must be 'summarize' or 'qa'")
    if not content:
        raise HTTPException(status_code=422, detail="content is required")

    key = json.dumps([task, hash(content), instructions])
    cached = cache_get(key)
    if cached:
        return JSONResponse({"answer": cached, "meta": {"cached": True}})

    messages = build_messages(task, content, instructions)

    # Run both providers concurrently. If one fails, we still try to return something.
    try:
        openai_task = providers.openai(messages)
        anthropic_task = providers.anthropic(messages)
        a, b = await asyncio.wait_for(asyncio.gather(openai_task, anthropic_task, return_exceptions=True), REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="upstream timeout")

    ans_a = a if isinstance(a, str) else ""
    ans_b = b if isinstance(b, str) else ""

    if not ans_a and not ans_b:
        raise HTTPException(status_code=502, detail="both providers failed")

    final, meta = fuse(ans_a, ans_b)
    cache_set(key, final)
    return JSONResponse({"answer": final, "meta": {**meta, "cached": False}})

# Healthcheck
@app.get("/healthz")
async def healthz():
    return {"ok": True}

# Shutdown cleanup
@app.on_event("shutdown")
async def shutdown_event():
    await providers.close()

# --- End of file ---
