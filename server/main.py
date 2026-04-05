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
