"""
inference.py — Baseline inference script for MetaModEnv.

Runs all 3 tasks sequentially using an LLM via OpenAI-compatible API.
Logs [START], [STEP], [END] for each task and prints a final summary.

Environment variables:
  API_BASE_URL   — OpenAI-compatible base URL (default: https://api.openai.com/v1)
  OPENAI_API_KEY — API key
  MODEL_NAME     — Model to use (default: gpt-4o-mini)
  HF_TOKEN       — HuggingFace token; used as fallback API key if OPENAI_API_KEY not set
  ENV_URL        — Base URL of the running FastAPI env server
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Use HF_TOKEN as fallback API key if OPENAI_API_KEY not set
# (for HuggingFace Inference API compatibility)
if not API_KEY and HF_TOKEN:
    API_KEY = HF_TOKEN

MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.6
TEMPERATURE = 0.0

TASKS = ["basic_moderation", "contextual_moderation", "adversarial_moderation"]

FALLBACK_ACTION = "KEEP"

# Maximum possible reward per task (used for per-task score normalization)
TASK_MAX_REWARDS: Dict[str, float] = {
    "basic_moderation": (
        8 * 0.20   # 8 correct removes
        + 7 * 0.15  # 7 correct keeps
        + 2 * 0.20  # 2 correct escalations
        + 3 * 0.15  # 3 correct labels
        + 0.10      # all escalations caught bonus
        + 0.15      # accuracy bonus
        + 0.05      # step budget bonus
    ),  # = 4.35
    "contextual_moderation": (
        30 * 0.20   # 30 correct decisions (upper bound)
        + 0.10 + 0.15 + 0.05
    ),  # = 6.30
    "adversarial_moderation": (
        50 * 0.20   # 50 correct decisions (upper bound)
        + 0.10 + 0.15 + 0.05
    ),  # = 10.30
}

SYSTEM_PROMPT = """You are an expert content moderator at Meta.
Your job is to review flagged content and make accurate moderation
decisions following Meta's Community Standards.

For each content item you review, you must respond with a JSON object
(and ONLY the JSON object — no prose, no markdown fences):
{
  "action_type": "REMOVE" | "KEEP" | "ESCALATE" | "ADD_WARNING_LABEL" | "REQUEST_CONTEXT",
  "target_item_id": "<the item id>",
  "policy_violated": "<policy name or null>",
  "reasoning": "<brief explanation>",
  "confidence": <0.0 to 1.0>
}

Policies to enforce:
- HATE_SPEECH: Content attacking people based on protected characteristics
- VIOLENCE: Graphic violence, threats, incitement
- MISINFORMATION: False information that could cause real-world harm
- SPAM: Unsolicited commercial content, scam links
- SELF_HARM: Content promoting or depicting self-harm or suicide
- HARASSMENT: Targeted harassment of individuals
- SEXUAL_CONTENT: Adult content in non-adult spaces

Decision guide:
- REMOVE: Clear, unambiguous policy violation
- KEEP: Does not violate policy; leave up
- ESCALATE: Urgent — self-harm signals, credible violent threats with specific targets
- ADD_WARNING_LABEL: Keep but flag — graphic-but-newsworthy, borderline health claims
- REQUEST_CONTEXT: Genuinely need more information before deciding

Always read the user_history context:
- is_satire_account=true + is_verified=true → likely satire, use caution before REMOVE
- is_verified=true + journalist context → ADD_WARNING_LABEL not REMOVE for graphic content
- account_age_days < 30 + prior_violations > 0 → higher risk account
- Multiple near-identical posts from new accounts → coordinated inauthentic behavior → REMOVE

Watch for policy circumvention — these MUST be REMOVE even though they use tricks:
- Deliberate misspellings (h@te, k*ll, 0 replacing o, 3 replacing e, 1 replacing i)
- Emoji sequences replacing violent verbs (🔪🔪, 🧹🧹) or derogatory nouns (🐀🐀🐀)
- Coded language and dogwhistles ("1488", "final solution" targeting a group, "it is what it is with X")
- Romanized or foreign-script slurs ("these foreign types", "these outsiders")
- Any content that is clearly hate speech when read normally, regardless of obfuscation

ESCALATE immediately — do not REMOVE or KEEP these:
- Any post signalling imminent self-harm: "planning my exit", "tonight is the night",
  "saying goodbye", "stockpiling [anything]", "nobody would notice if I disappeared",
  "made my decision", "I have it all planned" — even if phrased as casual venting
- Specific credible threats: named target + location + "I know their schedule"

Near-miss content that looks like a violation but is NOT — use KEEP:
- "final solution to climate change / budget deficit" — innocent use of phrase
- "kill it at the presentation / destroy the competition" — figurative language
- Immigrant/minority activists organizing or voting — protected political speech
- "execute our plan / dominate the market" — business language
"""


# ---------------------------------------------------------------------------
# Logging — EXACT format required by OpenEnv spec
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_str = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={action!r} "
        f"reward={reward:.4f} done={done} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    print(
        f"[END] success={success} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Env server calls
# ---------------------------------------------------------------------------

def env_reset(
    client: httpx.Client,
    task_name: str,
    session_id: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name, "session_id": session_id},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def env_step(
    client: httpx.Client,
    action_dict: Dict[str, Any],
    session_id: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{ENV_URL}/step",
        json={"action": action_dict, "session_id": session_id},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def env_state(
    client: httpx.Client,
    session_id: str,
) -> Dict[str, Any]:
    response = client.get(
        f"{ENV_URL}/state",
        params={"session_id": session_id},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(llm_client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM with current observation and return a parsed action dict."""
    current_item = obs.get("current_item", {})
    user_history = current_item.get("user_history", {})

    # Build recent-items context for coordinated behaviour detection
    reviewed_so_far = obs.get("reviewed_so_far", [])
    recent = reviewed_so_far[-5:] if reviewed_so_far else []
    recent_context = ""
    if recent:
        recent_lines = "\n".join(
            f"  - item {r.get('item_id', '?')}: action={r.get('action_taken', '?')}"
            for r in recent
        )
        recent_context = f"\nRecent decisions (last {len(recent)}):\n{recent_lines}\n"

    user_message = f"""Review the following flagged content item:

Item ID: {current_item.get('item_id', 'unknown')}
Platform: {current_item.get('platform', 'unknown')}
Content: {current_item.get('content_text', '')}

User History:
  Account age (days): {user_history.get('account_age_days', 'unknown')}
  Prior violations: {user_history.get('prior_violations', 0)}
  Follower count: {user_history.get('follower_count', 0)}
  Verified: {user_history.get('is_verified', False)}
  Satire account: {user_history.get('is_satire_account', False)}

Report count: {current_item.get('report_count', 0)}
Posted at: {current_item.get('timestamp', '')}

Queue position: {obs.get('queue_position', '?')}/{obs.get('queue_total', '?')}
Goal: {obs.get('goal', '')}
{recent_context}
Last action error (if any): {obs.get('last_action_error') or 'none'}

Respond with the JSON action object only."""

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(
            f"[WARN] JSON parse failed on step — using KEEP fallback. "
            f"Raw response was: {raw[:120]!r}",
            flush=True,
        )
        return {
            "action_type": FALLBACK_ACTION,
            "target_item_id": current_item.get("item_id", "unknown"),
            "policy_violated": None,
            "reasoning": "Failed to parse model response — using fallback action.",
            "confidence": 0.5,
        }
    except Exception as e:
        print(
            f"[ERROR] LLM call failed: {type(e).__name__}: {e} — using KEEP fallback. "
            f"Check API_BASE_URL, OPENAI_API_KEY, and MODEL_NAME.",
            flush=True,
        )
        return {
            "action_type": FALLBACK_ACTION,
            "target_item_id": current_item.get("item_id", "unknown"),
            "policy_violated": None,
            "reasoning": f"LLM call failed ({e}) — using fallback action.",
            "confidence": 0.5,
        }


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    llm_client: OpenAI,
    http_client: httpx.Client,
) -> Dict[str, Any]:
    # Fresh unique session per task run
    SESSION_ID = str(uuid.uuid4())

    log_start(task_name, ENV_URL, MODEL_NAME)

    result = env_reset(http_client, task_name, SESSION_ID)
    obs = result.get("observation")
    done = result.get("done", False)

    step_num = 0
    rewards: List[float] = []
    final_info: Dict[str, Any] = {}

    try:
        while obs and not done and step_num < MAX_STEPS:
            action_dict = call_llm(llm_client, obs)

            step_result = env_step(http_client, action_dict, SESSION_ID)

            reward_val = step_result.get("reward", {}).get("value", 0.0)
            done = step_result.get("done", False)
            error = step_result.get("info", {}).get("error")
            obs = step_result.get("observation")
            final_info = step_result.get("info", {})

            rewards.append(reward_val)
            step_num += 1

            log_step(step_num, action_dict.get("action_type", FALLBACK_ACTION), reward_val, done, error)
    finally:
        # Compute per-task normalized score
        max_reward = TASK_MAX_REWARDS.get(task_name, 10.0)
        raw_score = sum(rewards) / max_reward if max_reward > 0 else 0.0
        score = min(max(raw_score, 0.0), 1.0)

        # Prefer grader score from server if available
        grader_score = final_info.get("grader_score", 0.0)
        if grader_score and grader_score > 0:
            score = grader_score

        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success, step_num, score, rewards)

    return {
        "task": task_name,
        "steps": step_num,
        "score": score,
        "success": success,
        "rewards": rewards,
        "session_id": SESSION_ID,
        "info": final_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("WARNING: OPENAI_API_KEY and HF_TOKEN are both unset.", flush=True)

    llm_client = OpenAI(api_key=API_KEY or "placeholder", base_url=API_BASE_URL)

    with httpx.Client() as http_client:
        # Health ping
        try:
            health = http_client.get(f"{ENV_URL}/health", timeout=10)
            health.raise_for_status()
            print(f"[INFO] Environment server is up at {ENV_URL}", flush=True)
        except Exception as e:
            print(f"[ERROR] Cannot reach environment server at {ENV_URL}: {e}", flush=True)
            sys.exit(1)

        results: List[Dict[str, Any]] = []
        for task_name in TASKS:
            task_result = run_task(task_name, llm_client, http_client)
            results.append(task_result)
            print("", flush=True)

    # Final summary
    print("=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {r['task']:<30} score={r['score']:.4f}  steps={r['steps']}  [{status}]",
            flush=True,
        )
    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  Overall average score: {overall:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
