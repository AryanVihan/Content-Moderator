"""
MCP (Model Context Protocol) server for MetaModEnv.

Exposes the environment as an MCP server so Claude and other
MCP-compatible agents can use it directly.

Usage:
    python server/mcp_server.py

Or via Docker with MCP_MODE=true.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

mcp = FastMCP("MetaModEnv")


@mcp.tool()
async def reset_moderation_env(
    task_name: str = "basic_moderation",
    session_id: str = "mcp_session",
) -> dict:
    """
    Reset the content moderation environment and get the first item to review.

    Args:
        task_name: One of basic_moderation, contextual_moderation,
                   adversarial_moderation
        session_id: Unique session identifier

    Returns:
        Initial observation with the first flagged content item
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name, "session_id": session_id},
            timeout=30.0,
        )
        return response.json()


@mcp.tool()
async def moderate_content(
    action_type: str,
    target_item_id: str,
    reasoning: str,
    confidence: float,
    session_id: str = "mcp_session",
    policy_violated: Optional[str] = None,
) -> dict:
    """
    Submit a moderation decision for the current content item.

    Args:
        action_type: REMOVE | KEEP | ESCALATE | ADD_WARNING_LABEL |
                     REQUEST_CONTEXT
        target_item_id: ID of the content item being reviewed
        reasoning: Explanation for the decision
        confidence: Confidence score 0.0 to 1.0
        session_id: Session identifier (must match reset call)
        policy_violated: Required if action_type is REMOVE. One of:
                         HATE_SPEECH, VIOLENCE, MISINFORMATION, SPAM,
                         SELF_HARM, HARASSMENT, SEXUAL_CONTENT

    Returns:
        Next observation, reward received, and whether episode is done
    """
    action = {
        "action_type": action_type,
        "target_item_id": target_item_id,
        "policy_violated": policy_violated,
        "reasoning": reasoning,
        "confidence": confidence,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ENV_URL}/step",
            json={"action": action, "session_id": session_id},
            timeout=30.0,
        )
        return response.json()


@mcp.tool()
async def get_moderation_state(session_id: str = "mcp_session") -> dict:
    """
    Get the current state of the moderation episode.

    Args:
        session_id: Session identifier

    Returns:
        Full current episode state including all reviewed items and scores
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{ENV_URL}/state",
            params={"session_id": session_id},
            timeout=30.0,
        )
        return response.json()


@mcp.tool()
async def list_moderation_tasks() -> dict:
    """
    List all available moderation tasks with difficulty levels.

    Returns:
        List of tasks with names, descriptions, and difficulty ratings
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ENV_URL}/tasks", timeout=30.0)
        return {"tasks": response.json()}


if __name__ == "__main__":
    mcp.run()
