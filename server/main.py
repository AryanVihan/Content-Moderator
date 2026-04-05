"""FastAPI server — exposes OpenEnv endpoints for MetaModEnv."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.environment import (
    get_state,
    list_sessions,
    reset_episode,
    step_episode,
)
from server.models import Action, StepResult
from server.tasks import TASKS

ENABLE_WEB_INTERFACE = os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true"

app = FastAPI(
    title="MetaModEnv",
    description="OpenEnv — Meta Content Moderation Simulation Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Gradio UI if enabled
if ENABLE_WEB_INTERFACE:
    try:
        import gradio as gr
        from server.gradio_ui import build_ui
        ui = build_ui()
        app = gr.mount_gradio_app(app, ui, path="/ui")
    except ImportError:
        print("[MetaModEnv] WARNING: gradio not installed — ENABLE_WEB_INTERFACE ignored.", flush=True)


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    model = os.environ.get("MODEL_NAME", "not set")
    interface = os.environ.get("ENABLE_WEB_INTERFACE", "false")
    print(f"[MetaModEnv] Starting up", flush=True)
    print(f"[MetaModEnv] MODEL_NAME={model}", flush=True)
    print(f"[MetaModEnv] ENABLE_WEB_INTERFACE={interface}", flush=True)
    print(f"[MetaModEnv] MCP_MODE={os.environ.get('MCP_MODE', 'false')}", flush=True)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "basic_moderation"
    session_id: str = "default"


class StepRequest(BaseModel):
    action: Action
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Landing page — links to docs and available endpoints."""
    from fastapi.responses import HTMLResponse
    html = """<!DOCTYPE html>
<html>
<head>
  <title>MetaModEnv</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; color: #1a1a1a; }
    h1 { font-size: 1.8rem; }
    .badge { background:#2563eb; color:#fff; border-radius:4px; padding:2px 8px; font-size:.8rem; }
    .badge.green { background:#16a34a; }
    .badge.orange { background:#d97706; }
    .badge.red { background:#dc2626; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { text-align:left; padding:8px 12px; border-bottom:1px solid #e5e7eb; }
    th { background:#f9fafb; }
    a { color:#2563eb; text-decoration:none; }
    a:hover { text-decoration:underline; }
    code { background:#f3f4f6; padding:2px 6px; border-radius:3px; font-size:.9rem; }
    .section { margin-top: 2rem; }
  </style>
</head>
<body>
  <h1>🛡️ MetaModEnv</h1>
  <p>OpenEnv — Meta Content Moderation Simulation Environment</p>

  <div class="section">
    <h2>Quick Links</h2>
    <ul>
      <li><a href="/docs">Interactive API Docs (Swagger UI)</a></li>
      <li><a href="/health">Health Check</a></li>
      <li><a href="/tasks">Available Tasks</a></li>
      <li><a href="/state">Current State</a></li>
    </ul>
  </div>

  <div class="section">
    <h2>Tasks</h2>
    <table>
      <tr><th>Task</th><th>Difficulty</th><th>Queue</th><th>Expected Score</th></tr>
      <tr><td><code>basic_moderation</code></td><td><span class="badge green">EASY</span></td><td>20 items</td><td>0.80 – 0.95</td></tr>
      <tr><td><code>contextual_moderation</code></td><td><span class="badge orange">MEDIUM</span></td><td>30 items</td><td>0.50 – 0.70</td></tr>
      <tr><td><code>adversarial_moderation</code></td><td><span class="badge red">HARD</span></td><td>50 items</td><td>0.35 – 0.55</td></tr>
    </table>
  </div>

  <div class="section">
    <h2>API Usage</h2>
    <p>Start a session:</p>
    <pre><code>POST /reset
{"task_name": "basic_moderation", "session_id": "my_session"}</code></pre>
    <p>Submit a decision:</p>
    <pre><code>POST /step
{"action": {"action_type": "REMOVE", "target_item_id": "...",
 "policy_violated": "HATE_SPEECH", "reasoning": "...", "confidence": 0.9},
 "session_id": "my_session"}</code></pre>
  </div>

  <div class="section">
    <p style="color:#6b7280;font-size:.85rem;">
      v1.0.0 &nbsp;·&nbsp;
      <a href="https://github.com/AryanVihan/Content-Moderator">GitHub</a> &nbsp;·&nbsp;
      OpenEnv compliant
    </p>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Deployment ping — returns 200 OK."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest) -> StepResult:
    """
    Initialize (or re-initialize) an episode.
    Returns the first observation with an empty reward.
    """
    valid_tasks = list(TASKS.keys())
    if request.task_name not in valid_tasks:
        return StepResult(
            observation=None,
            reward={"value": 0.0, "breakdown": {}, "cumulative": 0.0},  # type: ignore
            done=True,
            info={
                "error": f"Unknown task '{request.task_name}'. "
                         f"Valid options: {valid_tasks}"
            },
        )
    return reset_episode(request.task_name, request.session_id)


@app.post("/step")
async def step(request: StepRequest) -> StepResult:
    """
    Submit a moderation action for the current item.
    Returns the next observation, reward, and done flag.
    """
    return step_episode(request.action, request.session_id)


@app.get("/state")
async def state(session_id: str = Query(default="default")) -> Dict[str, Any]:
    """Return the full current environment state for a session."""
    return get_state(session_id)


@app.get("/tasks")
async def list_tasks() -> List[Dict[str, Any]]:
    """Return all available tasks with descriptions."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "difficulty": t.difficulty,
            "queue_size": t.queue_size,
            "max_steps": t.max_steps,
            "expected_score_min": t.expected_score_min,
            "expected_score_max": t.expected_score_max,
        }
        for t in TASKS.values()
    ]


@app.get("/sessions")
async def sessions() -> Dict[str, Any]:
    """Return list of active session IDs."""
    return {"sessions": list_sessions()}
