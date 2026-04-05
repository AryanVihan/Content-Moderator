"""Gradio web interface for MetaModEnv — activated via ENABLE_WEB_INTERFACE=true."""
from __future__ import annotations

import os
from typing import Tuple

import gradio as gr
import httpx

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Tracks the current item_id across steps (state stored in Gradio State)


def reset_environment(
    task_name: str,
    session_id: str,
) -> Tuple[str, str, str, str]:
    """Reset the episode and return the first item observation."""
    try:
        response = httpx.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name, "session_id": session_id or "gradio_session"},
            timeout=30.0,
        )
        result = response.json()
    except Exception as e:
        return f"Error connecting to server: {e}", "Not started", "", ""

    obs = result.get("observation", {})
    item = obs.get("current_item", {})
    uh = item.get("user_history", {})

    obs_text = (
        f"**Goal:** {obs.get('goal', '')}\n\n"
        f"**Item ID:** `{item.get('item_id', '')}`\n\n"
        f"**Platform:** {item.get('platform', '')}\n\n"
        f"**Content:** {item.get('content_text', '')}\n\n"
        f"**Reports:** {item.get('report_count', 0)}\n\n"
        f"**Account age:** {uh.get('account_age_days', 0)} days\n\n"
        f"**Prior violations:** {uh.get('prior_violations', 0)}\n\n"
        f"**Verified:** {uh.get('is_verified', False)} | "
        f"**Satire account:** {uh.get('is_satire_account', False)}\n\n"
        f"**Queue:** {obs.get('queue_position', 0)}/{obs.get('queue_total', 0)}"
    )

    status = "Episode started. Step 0. Reward: 0.00"
    item_id = item.get("item_id", "")
    sid = session_id or "gradio_session"

    return obs_text, status, item_id, sid


def take_step(
    action_type: str,
    policy_violated: str,
    reasoning: str,
    confidence: float,
    session_id: str,
    current_item_id: str,
) -> Tuple[str, str, str]:
    """Submit a moderation decision and return the next observation."""
    if not current_item_id:
        return (
            "*No active episode. Click 'Reset / New Episode' first.*",
            "No active episode.",
            "",
        )

    action = {
        "action_type": action_type,
        "target_item_id": current_item_id,
        "policy_violated": policy_violated if policy_violated else None,
        "reasoning": reasoning or "No reasoning provided.",
        "confidence": float(confidence),
    }

    try:
        response = httpx.post(
            f"{ENV_URL}/step",
            json={"action": action, "session_id": session_id},
            timeout=30.0,
        )
        result = response.json()
    except Exception as e:
        return f"Error: {e}", "Step failed.", current_item_id

    obs = result.get("observation") or {}
    item = obs.get("current_item", {}) if obs else {}
    uh = item.get("user_history", {}) if item else {}
    reward_val = result.get("reward", {}).get("value", 0.0)
    cumulative = result.get("reward", {}).get("cumulative", 0.0)
    done = result.get("done", False)
    info = result.get("info", {})

    if done:
        grader_score = info.get("grader_score", 0.0)
        normalized = info.get("normalized_score", 0.0)
        obs_text = (
            f"## Episode Complete!\n\n"
            f"**Grader score:** {grader_score:.4f}\n\n"
            f"**Normalized score:** {normalized:.4f}\n\n"
            f"**Total steps:** {info.get('steps_used', 0)}\n\n"
            f"**Items reviewed:** {info.get('items_reviewed', 0)}"
        )
        status = f"DONE | Final reward: {reward_val:+.2f} | Cumulative: {cumulative:.2f}"
        return obs_text, status, ""

    obs_text = (
        f"**Item ID:** `{item.get('item_id', '')}`\n\n"
        f"**Platform:** {item.get('platform', '')}\n\n"
        f"**Content:** {item.get('content_text', '')}\n\n"
        f"**Reports:** {item.get('report_count', 0)}\n\n"
        f"**Account age:** {uh.get('account_age_days', 0)} days\n\n"
        f"**Prior violations:** {uh.get('prior_violations', 0)}\n\n"
        f"**Verified:** {uh.get('is_verified', False)} | "
        f"**Satire account:** {uh.get('is_satire_account', False)}\n\n"
        f"**Queue:** {obs.get('queue_position', 0)}/{obs.get('queue_total', 0)}"
    )

    status = f"Reward this step: {reward_val:+.2f} | Cumulative: {cumulative:.2f} | Done: {done}"
    if obs.get("last_action_error"):
        status += f" | Error: {obs['last_action_error']}"

    next_item_id = item.get("item_id", "")
    return obs_text, status, next_item_id


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MetaModEnv — Content Moderation Sandbox") as demo:
        gr.Markdown("# MetaModEnv — Meta Content Moderation Environment")
        gr.Markdown(
            "An OpenEnv environment simulating Meta's content moderation queue. "
            "Review flagged content and make policy decisions."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                task_dropdown = gr.Dropdown(
                    choices=[
                        "basic_moderation",
                        "contextual_moderation",
                        "adversarial_moderation",
                    ],
                    value="basic_moderation",
                    label="Select Task",
                )
                session_input = gr.Textbox(
                    value="gradio_session_1",
                    label="Session ID",
                )
                reset_btn = gr.Button("Reset / New Episode", variant="primary")

                gr.Markdown("### Take Action")
                action_type = gr.Radio(
                    choices=["REMOVE", "KEEP", "ESCALATE", "ADD_WARNING_LABEL", "REQUEST_CONTEXT"],
                    value="KEEP",
                    label="Action",
                )
                policy_input = gr.Dropdown(
                    choices=[
                        "",
                        "HATE_SPEECH",
                        "VIOLENCE",
                        "MISINFORMATION",
                        "SPAM",
                        "SELF_HARM",
                        "HARASSMENT",
                        "SEXUAL_CONTENT",
                    ],
                    value="",
                    label="Policy Violated (only for REMOVE)",
                )
                reasoning_input = gr.Textbox(
                    placeholder="Why are you making this decision?",
                    label="Reasoning",
                    lines=2,
                )
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                    label="Confidence",
                )
                step_btn = gr.Button("Submit Decision", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("### Current Content Item")
                observation_display = gr.Markdown(
                    value="*Click 'Reset / New Episode' to start.*"
                )
                status_display = gr.Textbox(label="Step Status", interactive=False)
                current_item_id = gr.Textbox(visible=False, value="")

        reset_btn.click(
            fn=reset_environment,
            inputs=[task_dropdown, session_input],
            outputs=[observation_display, status_display, current_item_id, session_input],
        )

        step_btn.click(
            fn=take_step,
            inputs=[
                action_type,
                policy_input,
                reasoning_input,
                confidence_slider,
                session_input,
                current_item_id,
            ],
            outputs=[observation_display, status_display, current_item_id],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7861)
