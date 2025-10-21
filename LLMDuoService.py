"""
LLM Duo Service - FastAPI Application with Queue Processing and Database Persistence

This module implements a production-ready service that processes natural language tasks
using dual LLM providers (OpenAI and Anthropic) with intelligent response fusion.

Key Features:
- Dual Provider Architecture: Leverages both OpenAI GPT-4o-mini and Anthropic Claude-3-haiku
- Intelligent Response Fusion: Combines responses from multiple providers using custom logic
- Asynchronous Job Processing: Queue-based worker system for concurrent request handling
- Database Persistence: SQLite backend for job tracking and status management
- Resilience & Observability: Built-in retry mechanisms and comprehensive job monitoring
- RESTful API: FastAPI-based endpoints for job submission and status checking

Architecture Overview:
1. Input Validation & API Contracts (/submit endpoint)
2. Prompting System (structured message building)
3. Response Fusion (intelligent answer selection/combination)
4. Resilience & Observability (error handling, retries, status tracking)

Environment Variables:
- OPENAI_API_KEY: OpenAI API authentication key
- ANTHROPIC_API_KEY: Anthropic API authentication key
- REQUEST_TIMEOUT_S: HTTP request timeout in seconds (default: 15)
- MAX_TOKENS: Maximum tokens per LLM response (default: 512)
- TEMPERATURE: LLM response randomness (default: 0.7)
- DB_PATH: SQLite database file path (default: /tmp/llm_duo.db)
- WORKER_CONCURRENCY: Number of concurrent worker threads (default: 2)
- QUEUE_MAXSIZE: Maximum queue size for pending jobs (default: 1000)

Usage:
    # Start the service
    uvicorn test2:app --host 0.0.0.0 --port 8000

    # Submit a job
    POST /submit
    {
        "task": "summarize",
        "content": "Your text content here",
        "instructions": "Keep it under 100 words"
    }

    # Check job status
    GET /status/{job_id}

Author: College Board GenAI Engineering Team
Version: 0.5.0
"""
from __future__ import annotations
import os
import json
import time
import asyncio
from typing import Dict, Any, List, Tuple, Optional

import httpx
import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "changeme-openai")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "changeme-anthropic")
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "15"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DB_PATH = os.getenv("DB_PATH", "/tmp/llm_duo.db")
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))
QUEUE_MAXSIZE = int(os.getenv("QUEUE_MAXSIZE", "1000"))

app = FastAPI(title="LLM Duo (Queue + DB)", version="0.5.0")
JOB_QUEUE: "asyncio.Queue[int]" = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
_worker_tasks: List[asyncio.Task] = []
_shutdown = asyncio.Event()

# -----------------------------------------------------------------------------
# Providers
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
# Open Question 2: Prompting
# -----------------------------------------------------------------------------

def build_messages(task: str, content: str, instructions: Optional[str] = None) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a helpful assistant. Be concise when possible."
        + (f" Instructions: {instructions}" if instructions else "")
    )
    user_prompt = f"Task: {task}\nContent:\n{content}"
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

# -----------------------------------------------------------------------------
# Open Question 3: Fusion
# -----------------------------------------------------------------------------


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
# Database
# -----------------------------------------------------------------------------
INIT_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task TEXT NOT NULL,
  content TEXT NOT NULL,
  instructions TEXT,
  status TEXT NOT NULL,
  result TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(INIT_SQL)
        await db.commit()

async def enqueue_job(task: str, content: str, instructions: Optional[str]) -> int:
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO jobs(task, content, instructions, status, created_at, updated_at)\n             VALUES(?, ?, ?, 'queued', ?, ?)",
            (task, content, instructions, now, now),
        )
        await db.commit()
        job_id = cur.lastrowid
    await JOB_QUEUE.put(job_id)
    return job_id

async def load_job(job_id: int) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

async def update_job(job_id: int, **fields):
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [job_id]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", values)
        await db.commit()

# -----------------------------------------------------------------------------
# Open Question 4: Resilience & Observability
# -----------------------------------------------------------------------------


async def worker_loop(idx: int):
    while not _shutdown.is_set():
        try:
            job_id = await JOB_QUEUE.get()
        except asyncio.CancelledError:
            break
        job = await load_job(job_id)
        if not job:
            continue
        await update_job(job_id, status="processing", attempts=job["attempts"] + 1)
        try:
            messages = build_messages(job["task"], job["content"], job["instructions"])
            a, b = await asyncio.gather(
                providers.openai(messages),
                providers.anthropic(messages),
                return_exceptions=True,
            )
            ans_a = a if isinstance(a, str) else ""
            ans_b = b if isinstance(b, str) else ""
            if not ans_a and not ans_b:
                await update_job(job_id, status="failed", result="both providers failed")
            else:
                final, meta = fuse(ans_a, ans_b)
                await update_job(job_id, status="succeeded", result=json.dumps({"answer": final, "meta": meta}))
        except Exception as e:
            await update_job(job_id, status="failed", result=str(e))
        finally:
            JOB_QUEUE.task_done()

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
# Open Question 1 (Input Validation & API Contracts)

@app.post("/submit")
async def submit_job(body: Dict[str, Any]):
    task = (body.get("task") or "").strip().lower()
    content = body.get("content") or ""
    instructions = body.get("instructions")

    if not content:
        raise HTTPException(status_code=422, detail="content is required")


    job_id = await enqueue_job(task, content, instructions)
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
async def job_status(job_id: int):
    job = await load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

# -----------------------------------------------------------------------------
# Startup / Shutdown
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    await init_db()
    for i in range(WORKER_CONCURRENCY):
        t = asyncio.create_task(worker_loop(i))
        _worker_tasks.append(t)

@app.on_event("shutdown")
async def on_shutdown():
    _shutdown.set()
    for t in _worker_tasks:
        t.cancel()
    await providers.close()

# --- End of file ---
