---
title: MetaModEnv
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - content-moderation
  - nlp
  - trust-safety
  - meta
---

# MetaModEnv — Meta Content Moderation OpenEnv

## 1. Environment Description & Motivation

Content moderation at scale is one of the hardest problems in AI safety. Meta's moderation teams review millions of posts daily, making split-second decisions that balance free expression against user safety. A single missed escalation can mean a life lost; a single false positive removes legitimate speech.

**MetaModEnv** simulates this environment. An AI agent takes the role of a human moderator working through a queue of flagged content — posts, comments, marketplace listings, and forwarded messages from Facebook, Instagram, WhatsApp, and Marketplace. The agent must apply Meta's Community Standards policies accurately, handle ambiguous edge cases using context, and prioritize urgent escalations (self-harm, credible threats).

The environment tests:
- **Policy judgment** — applying rules correctly across clear and ambiguous cases
- **Context sensitivity** — same text can be fine or a violation depending on account context
- **Adversarial robustness** — detecting evasion via misspellings, emoji substitution, coded language
- **Priority awareness** — never missing urgent escalation items buried in the queue

---

## 2. Quick Start

### Docker

```bash
# Build
docker build -t meta-mod-env .

# Run
docker run -p 7860:7860 meta-mod-env

# Verify
curl http://localhost:7860/health
```

### Run inference

```bash
export OPENAI_API_KEY="your-key-here"
export ENV_URL="http://localhost:7860"
export MODEL_NAME="gpt-4o-mini"   # or any OpenAI-compatible model

python inference.py
```

---

## 3. Action Space

| Action | Description |
|--------|-------------|
| `REMOVE` | Content violates policy — remove from platform. Must set `policy_violated`. |
| `KEEP` | Content is fine — leave it up with no action. |
| `ESCALATE` | Urgent: imminent self-harm, credible specific threat — route to crisis team immediately. |
| `ADD_WARNING_LABEL` | Keep content but attach a warning label (graphic/newsworthy, health claims). |
| `REQUEST_CONTEXT` | Need more information before making a confident decision. Neutral (0 reward). |

---

## 4. Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `goal` | string | Natural language task goal for this episode. |
| `current_item.item_id` | string | Unique ID of the flagged item. |
| `current_item.platform` | string | `facebook` / `instagram` / `whatsapp` / `marketplace` |
| `current_item.content_text` | string | The actual flagged text. |
| `current_item.user_history.account_age_days` | int | Account age in days. |
| `current_item.user_history.prior_violations` | int | Past policy violations. |
| `current_item.user_history.follower_count` | int | Follower/friend count. |
| `current_item.user_history.is_verified` | bool | Verified badge status. |
| `current_item.user_history.is_satire_account` | bool | Account explicitly marked as parody/satire. |
| `current_item.report_count` | int | Number of user reports. |
| `current_item.timestamp` | string | ISO 8601 post timestamp. |
| `queue_position` | int | Current position in queue. |
| `queue_total` | int | Total items in queue. |
| `reviewed_so_far` | list | History of all prior decisions with rewards. |
| `context.task` | string | Task name. |
| `context.difficulty` | string | `EASY` / `MEDIUM` / `HARD` |
| `context.steps_remaining` | int | Steps left in budget. |
| `last_action_error` | string or null | Error from previous action, if any. |

---

## 5. Task Descriptions

| Task | Difficulty | Queue Size | Max Steps | Expected Score |
|------|-----------|-----------|-----------|----------------|
| `basic_moderation` | EASY | 20 | 25 | 0.80 – 0.95 |
| `contextual_moderation` | MEDIUM | 30 | 40 | 0.50 – 0.70 |
| `adversarial_moderation` | HARD | 50 | 60 | 0.35 – 0.55 |

### basic_moderation
Clear-cut decisions: 8 obvious violations, 7 safe posts, 3 borderline (warning labels), 2 urgent escalations. No ambiguity.

### contextual_moderation
Context determines the verdict: satire accounts, verified journalists, reclaimed language, technically-true-but-misleading content, and coordinated inauthentic behavior signals across items.

### adversarial_moderation
Designed to trick: deliberate misspellings (`h@te`, `k*ll`), emoji substitutions, Romanized cross-language slurs, viral misinformation with millions of shares, political speech edge cases, 10-account brigading campaign, and 5 urgent escalations buried in the queue.

---

## 6. Reward Function

### Per-Step Rewards

| Event | Reward |
|-------|--------|
| Correctly REMOVE a genuine violation | +0.20 |
| Correctly KEEP safe content | +0.15 |
| Correctly ESCALATE urgent content | +0.20 |
| Correctly ADD_WARNING_LABEL to borderline content | +0.15 |
| High confidence (>0.8) on correct decision | +0.05 |
| False positive REMOVE (safe content removed) | -0.15 |
| **Missed ESCALATION on urgent item** | **-0.25** |
| KEEP on a clear violation | -0.10 |
| Low confidence (<0.4) on obvious decision | -0.05 |
| Wrong `policy_violated` field on REMOVE | -0.20 |

### End-of-Episode Bonuses

| Event | Bonus |
|-------|-------|
| Overall accuracy > 85% | +0.15 |
| All urgent escalations caught | +0.10 |
| Completed within step budget | +0.05 |

All per-step rewards clamp to `[-1.0, 1.0]`. Final grader score normalized to `[0.0, 1.0]`.

---

## 7. Baseline Scores

| Task | Baseline (gpt-4o-mini) |
|------|----------------------|
| `basic_moderation` | ~0.85 |
| `contextual_moderation` | ~0.60 |
| `adversarial_moderation` | ~0.45 |

---

## 8. Local Setup (without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the environment server
uvicorn server.main:app --host 0.0.0.0 --port 7860 --reload

# 4. In a separate terminal, run inference
export OPENAI_API_KEY="your-key"
python inference.py
```

---

## 9. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(required)_ | API key for LLM calls in `inference.py` |
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model name |
| `ENV_URL` | `http://localhost:7860` | URL of the running environment server |
| `HF_TOKEN` | _(optional)_ | HuggingFace token for HF Spaces deployment |

---

## 10. OpenEnv Compliance

This environment implements the full OpenEnv interface:

- **Typed Pydantic models**: `Observation`, `Action`, `Reward`, `StepResult`
- **Required endpoints**: `POST /reset`, `POST /step`, `GET /state`, `GET /health`
- **Session management**: Multiple concurrent sessions via `session_id` parameter
- **Reproducible data**: `random.seed(42)` — identical queue on every run
- **No external dependencies**: Pure Python logic; no ML models, no external APIs in server
- **Per-step + episode rewards**: Granular reward signal for RL training
- **Graceful error handling**: Invalid actions return `last_action_error` in observation
- **openenv.yaml**: Full spec metadata including all tasks, action/observation space, reward range

---

## 11. API Reference

### POST /reset
```json
{
  "task_name": "basic_moderation",
  "session_id": "my_session"
}
```
Returns `StepResult` with initial observation.

### POST /step
```json
{
  "action": {
    "action_type": "REMOVE",
    "target_item_id": "basic_001",
    "policy_violated": "HATE_SPEECH",
    "reasoning": "Clear hate speech targeting a protected group.",
    "confidence": 0.95
  },
  "session_id": "my_session"
}
```
Returns `StepResult` with next observation, reward, done flag.

### GET /state?session_id=my_session
Returns full current episode state including all reviewed items.

### GET /health
Returns `{"status": "ok"}`.

### GET /tasks
Returns list of all available tasks with descriptions.
