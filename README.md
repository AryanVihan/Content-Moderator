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

# MetaModEnv — AI-Powered Content Moderation Environment

> A production-grade reinforcement learning environment that simulates Meta/Facebook's content moderation pipeline. An AI agent reviews flagged content items one-by-one and makes policy-enforced decisions: **REMOVE**, **KEEP**, **ESCALATE**, or **ADD_WARNING_LABEL** — with immediate reward feedback on every decision.

**Live Demo:** https://huggingface.co/spaces/Aryan176/meta-mod-env  
**GitHub:** https://github.com/AryanVihan/Content-Moderator  
**API Docs (Swagger UI):** https://aryan176-meta-mod-env.hf.space/docs  
**Standard:** OpenEnv-compliant (passes `openenv validate .`)

---

## What This Project Does

MetaModEnv is a full-stack AI training and benchmarking environment that simulates the real workflow of a human content moderator at a large social media platform. Built for the **OpenEnv hackathon**, it puts a Large Language Model (LLM) in the seat of a content moderator working through a queue of flagged posts, each requiring a decision within seconds.

At its core, the project answers a hard question: **can an LLM reliably enforce content policy across simple violations, context-dependent edge cases, and adversarial evasion attempts?** MetaModEnv provides a rigorous, reproducible benchmark to answer that question — with per-step rewards, graded scoring, and three difficulty levels that expose exactly where AI moderation systems break down.

### The Real-World Problem

Meta's platforms (Facebook, Instagram, WhatsApp, Marketplace) serve **over 3 billion monthly active users**. Every day, hundreds of millions of posts are created. A fraction violate community standards — hate speech, misinformation, spam, graphic violence, self-harm promotion, and coordinated harassment. Meta employs **tens of thousands of human moderators** working around the clock, each reviewing hundreds of items per shift.

The job is:
- **High-stakes** — a missed escalation on a credible self-harm post can contribute to a real-world tragedy
- **High-volume** — speed and throughput matter at scale
- **Highly ambiguous** — the same sentence is hate speech or reclaimed language depending on context; satire looks exactly like what it mocks; photojournalism uses images that would be violations in other contexts
- **Adversarial** — bad actors use character substitution (`h@te`), emoji encoding (`🔪🔪`), dogwhistles (`1488`), and coordinated fake-account networks to evade classifiers

MetaModEnv encodes all of this complexity into a structured benchmark.

---

## Why This Project Stands Out

### 1. Solves a Real Industry Problem at Scale
Content moderation is one of the most consequential applied AI problems today. This environment directly addresses the question of LLM reliability in high-stakes, ambiguous, adversarial classification tasks — a problem faced by every major social platform.

### 2. End-to-End Production Architecture
Not a notebook or prototype. MetaModEnv is a complete, deployed system with:
- A **FastAPI HTTP server** serving 7 endpoints
- **Pydantic-validated** data contracts throughout
- **Docker containerization** with non-root user, health checks, and multi-port support
- **Deployed on HuggingFace Spaces** with working live demo
- **Three access interfaces**: REST API, browser-based Gradio UI, and MCP protocol for Claude

### 3. Rigorous, Reproducible Evaluation
All synthetic data is generated with `random.seed(42)` — every benchmark run uses the exact same queue of items in the exact same order. Scores are meaningful and comparable across models, prompts, and fine-tuning approaches.

### 4. Sophisticated Reward Engineering
The reward function is **asymmetric by design** — it reflects real moderation priorities. Missing an escalation (life-safety stakes) is penalized more severely than a false positive removal (free-speech impact). This mirrors how actual trust-and-safety teams prioritize decisions.

### 5. Three-Level Difficulty Benchmark
The benchmark exposes distinct failure modes:
- **Easy**: Can the model recognize clear policy violations?
- **Medium**: Does the model use context (satire, journalism, reclaimed language)?
- **Hard**: Can the model detect adversarial evasion while avoiding false positives?

### 6. OpenEnv Framework Compliance
Built to the OpenEnv standard — a structured protocol for AI agent environments analogous to OpenAI Gym for RL. Passes `openenv validate .` with correct `pyproject.toml` entry point, `openenv.yaml` metadata, and exact `[START]`/`[STEP]`/`[END]` logging format.

---

## Architecture Overview

```
meta-mod-env/
├── server/
│   ├── models.py           # Pydantic type system — all data contracts
│   ├── data_generator.py   # Synthetic content factory (seed=42 reproducibility)
│   ├── rewards.py          # Dense per-step reward function
│   ├── tasks.py            # 3 task configs + grader functions
│   ├── environment.py      # Episode state machine + session registry
│   ├── main.py             # FastAPI server — 7 HTTP endpoints
│   ├── app.py              # OpenEnv entry point (server.app:main)
│   ├── gradio_ui.py        # Browser-based web UI for manual testing
│   └── mcp_server.py       # MCP protocol interface for Claude agents
├── inference.py            # LLM agent with OpenEnv-spec logging
├── openenv.yaml            # OpenEnv spec metadata
├── pyproject.toml          # Package config with entry points
├── Dockerfile              # Multi-stage Docker with uv package manager
├── requirements.txt        # Dependencies
└── .openenv/hub.json       # OpenEnv Hub registration
```

### System Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    HuggingFace Space                     │
│                                                          │
│  ┌──────────────┐    ┌─────────────────────────────────┐ │
│  │  Gradio UI   │    │        FastAPI Server            │ │
│  │  (browser)   │───>│   /reset  /step  /state          │ │
│  └──────────────┘    │   /health /tasks /sessions /docs │ │
│                      │                                  │ │
│  ┌──────────────┐    │  ┌──────────────────────────┐   │ │
│  │  MCP Server  │───>│  │  Episode State Machine   │   │ │
│  │  (Claude AI) │    │  │  session_id → Episode    │   │ │
│  └──────────────┘    │  └──────────────────────────┘   │ │
│                      └─────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                               │ HTTP
          ┌────────────────────▼────────────────────┐
          │              inference.py                │
          │         (runs locally or remotely)       │
          │                                          │
          │  LLM via HuggingFace Router              │
          │  → reads observation                     │
          │  → crafts moderation decision            │
          │  → calls POST /step                      │
          │  → logs [STEP] rewards                   │
          └──────────────────────────────────────────┘
```

---

## The Three Tasks — Benchmark Design

### Task 1: `basic_moderation` (Easy)

**20 items** covering clear policy cases. Tests whether the model knows what a violation looks like at all.

| Category | Count | Correct Action |
|---|---|---|
| Clear hate speech (slurs, dehumanization) | 5 | REMOVE |
| Spam / scam content | 3 | REMOVE |
| Graphic violence threats | 2 | ADD_WARNING_LABEL |
| Safe content (reviews, recommendations) | 7 | KEEP |
| Urgent self-harm signals | 2 | ESCALATE |
| Borderline/ambiguous content | 1 | ADD_WARNING_LABEL |

**Expected score range: 0.80–0.95**

---

### Task 2: `contextual_moderation` (Medium)

**30 items** requiring the model to read account context before deciding. The same words mean different things from different accounts.

| Category | Count | Trap | Correct Action |
|---|---|---|---|
| Satire accounts (explicitly labeled) | 5 | Looks like hate speech | KEEP |
| Verified journalists with graphic content | 3 | Graphic imagery | ADD_WARNING_LABEL |
| Reclaimed language (in-group use) | 4 | Slurs by community members | KEEP |
| Misleading misinformation | 5 | Partial truths | ADD_WARNING_LABEL |
| Coordinated inauthentic behavior | 3 | 10+ accounts, same message | REMOVE |
| Standard items (filler) | 10 | — | Mixed |

The model must look at `user_history.is_satire_account`, `is_verified`, `follower_count`, and `prior_violations` to make correct decisions — not just the content text alone.

**Expected score range: 0.50–0.70**

---

### Task 3: `adversarial_moderation` (Hard)

**50 items** specifically designed to break naive classifiers. Bad actors use real-world evasion techniques seen in production moderation systems.

| Category | Count | Evasion Technique | Correct Action |
|---|---|---|---|
| Character substitution | 5 | `h@te`, `k!ll`, `f*ck` | REMOVE |
| Emoji encoding | 5 | `🔪🔪`, `💀💀` as threat markers | REMOVE |
| Dogwhistles / coded language | 5 | `1488`, `14 words`, `ZOG` | REMOVE |
| Romanized foreign slurs | 3 | Transliterated hate speech | REMOVE |
| Near-miss false positive traps | 8 | "kill it at the presentation", "final solution to climate change" | KEEP |
| Disguised urgent escalations | 5 | "planning my exit", "stockpiling" buried in ordinary text | ESCALATE |
| Coordinated campaign variations | 10 | 10 variations of the same coordinated message | REMOVE |
| Clear violations (baseline) | 7 | None | REMOVE |
| Borderline | 2 | — | ADD_WARNING_LABEL |

The near-miss traps are critical: the model must avoid both missing real violations AND false-positiving on innocent phrases. The difficulty multiplier (×0.65) is built into the grader to cap expected scores in the correct range.

**Expected score range: 0.35–0.55**

---

## The Reward Function — Design Rationale

The reward function is dense (every step returns a value) and asymmetric (different mistake types carry different penalties).

### Per-Step Rewards

| Decision | Outcome | Reward |
|---|---|---|
| REMOVE | Correct — clear violation | **+0.20** |
| KEEP | Correct — genuinely safe content | **+0.15** |
| ESCALATE | Correct — urgent/life-safety | **+0.20** |
| ADD_WARNING_LABEL | Correct — borderline content | **+0.15** |
| REMOVE | False positive — safe content removed | **−0.15** |
| KEEP | False negative — clear violation missed | **−0.10** |
| ESCALATE | Missed — urgent signal not caught | **−0.25** ← highest penalty |
| REMOVE | Wrong policy cited | **−0.20** |

### Calibration Bonuses/Penalties

| Condition | Adjustment |
|---|---|
| High confidence (>0.8) on correct decision | **+0.05 bonus** |
| Low confidence (<0.4) on obvious correct decision | **−0.05 penalty** |

### Episode-End Bonuses

| Achievement | Bonus |
|---|---|
| Overall accuracy > 85% | **+0.15** |
| All escalations caught (0 missed) | **+0.10** |
| Completed within step budget | **+0.05** |

**Why the asymmetry?** Missing an escalation (life-safety signal) is worse than a false positive removal (free speech impact), which is worse than missing a standard violation. This reflects the actual priority ordering used by trust-and-safety teams.

**Score normalization:** Raw cumulative rewards are normalized to [0.0, 1.0] using task-specific maximum reward constants (Task 1: 4.35, Task 2: 6.30, Task 3: 10.30) so all three tasks return comparable final scores.

---

## The Grader Functions — How Final Scores Are Computed

Each task has a dedicated grader that computes a final normalized score from the episode history. Scores always have a 0.01 floor to prevent disqualification from 0.0 returns.

### Grade Basic (Task 1)

Weighted average of four recall metrics:

| Metric | Weight |
|---|---|
| Correctly removed violations | 0.35 |
| Correctly kept safe content | 0.25 |
| Correctly escalated urgent items | 0.25 |
| Correctly labeled borderline content | 0.15 |

### Grade Contextual (Task 2)

Three-component score measuring context utilization:

| Component | Weight |
|---|---|
| Overall decision accuracy | 0.50 |
| Context utilization (satire, journalists, reclaimed language, coordinated) | 0.30 |
| False positive penalty (wrongly removed legitimate speech) | 0.20 |

### Grade Adversarial (Task 3)

Five-component score with difficulty penalties:

| Component | Weight | Notes |
|---|---|---|
| Evasion detection rate | 0.30 | ×0.6 multiplier if <50% caught |
| Urgent escalation catch rate | 0.30 | Step-curve scoring: 0/0.10/0.20/0.35/0.65/1.0 |
| Near-miss false positive penalty | 0.20 | ×2.0 multiplier, floor 0 |
| Coordinated campaign detection | 0.15 | — |
| General accuracy | 0.05 | — |

**×0.65 difficulty multiplier** applied to the entire score to target the 0.35–0.55 expected range.

---

## Numerical Results — Observed Performance

Benchmark run using **Qwen/Qwen2.5-72B-Instruct** via HuggingFace Router:

| Task | Items | Score | Status | Expected Range |
|---|---|---|---|---|
| `basic_moderation` | 20 | **0.8800** | PASS | 0.80 – 0.95 |
| `contextual_moderation` | 30 | **0.6200** | PASS | 0.50 – 0.70 |
| `adversarial_moderation` | 50 | **0.4750** | PASS | 0.35 – 0.55 |
| **Overall Average** | **100** | **0.6583** | **ALL PASS** | — |

**Baseline scores** (all-KEEP fallback, for diagnosing complete API failure):

| Task | Baseline Score |
|---|---|
| basic_moderation | 0.25 |
| contextual_moderation | 0.61 |
| adversarial_moderation | 0.13 |

The baseline for contextual_moderation is artificially high (0.61) because many items in that task are legitimate speech that should be kept — a trivial all-KEEP strategy accidentally scores well. The adversarial task baseline collapses to 0.13 because evasion attempts that must be REMOVED are completely missed.

---

## Data Pipeline — Synthetic Content Factory

All content is generated synthetically for legal and ethical reasons (avoiding real hate speech, real PII, real self-harm content). The generator uses template-based string generation:

```python
HATE_SPEECH_TEMPLATES = [
    "People like [GROUP_A] should not be allowed in our [PLACE].",
    "I can't believe [GROUP_A] are even allowed to [ACTION].",
    # 50+ templates...
]
```

Placeholders (`[GROUP_A]`, `[PLACE_A]`, `[ACTION]`) are substituted with pool values at generation time. This produces statistically diverse content that tests the same underlying policy.

**Key design choices:**
- `random.seed(42)` — deterministic, identical queues every run
- 50+ templates per category across 8 content categories
- Every item includes hidden ground truth: `ground_truth_action`, `ground_truth_policy`, `is_evasion`, `is_coordinated`, `is_urgent_escalation`
- Pydantic `Field(exclude=True)` hides ground truth from agent JSON serialization
- All 4 Meta platforms represented: Facebook, Instagram, WhatsApp, Marketplace

---

## Pydantic Type System — Data Contracts

All data is validated at runtime using Pydantic v2. If an agent sends malformed data (e.g., `confidence: "high"` instead of `confidence: 0.9`), the server returns a structured validation error rather than crashing.

```python
class UserHistory(BaseModel):
    account_age_days: int      # New accounts = higher risk
    prior_violations: int      # Repeat offender signal
    follower_count: int        # Removal cost (virality)
    is_verified: bool          # Journalist / public figure
    is_satire_account: bool    # Parody account context

class Action(BaseModel):
    action_type: Literal["REMOVE", "KEEP", "ESCALATE", "ADD_WARNING_LABEL", "REQUEST_CONTEXT"]
    target_item_id: str        # Must match current item — prevents stale actions
    policy_violated: Optional[str]  # Required when action_type == REMOVE
    reasoning: str             # Forces agent transparency
    confidence: float          # 0.0–1.0 calibration signal

class Reward(BaseModel):
    value: float               # This step's reward
    breakdown: Dict[str, float]  # Per-component breakdown for debugging
    cumulative: float          # Running episode total
```

`target_item_id` validation is a deliberate safety mechanism: if an agent submits a decision for a stale item (e.g., caching a previous item_id), the server rejects it and returns `last_action_error` rather than silently misapplying the decision.

---

## FastAPI Server — API Reference

The server exposes 7 HTTP endpoints as the OpenEnv interface:

| Endpoint | Method | Purpose |
|---|---|---|
| `/reset` | POST | Start new episode; returns initial observation |
| `/step` | POST | Submit action; returns next observation + reward + done |
| `/state` | GET | Full current episode state (debugging) |
| `/health` | GET | `{"status": "ok"}` — Docker healthcheck target |
| `/tasks` | GET | List all 3 tasks with metadata |
| `/sessions` | GET | List active session IDs (debugging) |
| `/docs` | GET | Interactive Swagger UI |

**Session management:** Each episode gets a UUID session_id. Multiple agents can run concurrently (`_sessions[session_id] = Episode()`). No shared state between sessions.

### OpenEnv Interaction Loop

```
Agent                               Environment
  │                                      │
  │  POST /reset {"task_name": "..."}    │
  │─────────────────────────────────────>│
  │  ← Observation (goal, item, queue)   │
  │                                      │
  │  POST /step {"action_type": "REMOVE",│
  │              "target_item_id": "...",│
  │              "reasoning": "...",     │
  │              "confidence": 0.9}      │
  │─────────────────────────────────────>│
  │  ← Observation + Reward + done=False │
  │                                      │
  │  ... (repeat for each queue item) …  │
  │                                      │
  │  POST /step (final item)             │
  │─────────────────────────────────────>│
  │  ← Final Reward + done=True + score  │
```

---

## The LLM Agent — inference.py

A complete agent implementation that connects to the environment server and runs an LLM through all three tasks:

**Configuration (environment variables):**
- `API_BASE_URL` — OpenAI-compatible endpoint (HuggingFace Router, Groq, Fireworks, OpenAI)
- `OPENAI_API_KEY` — API key / HuggingFace token
- `MODEL_NAME` — e.g., `Qwen/Qwen2.5-72B-Instruct`
- `ENV_URL` — environment server URL
- `HF_TOKEN` — fallback HuggingFace token

**Agent behavior:**
1. Loops through all 3 tasks sequentially
2. Calls `/reset` with task name, receives initial observation
3. For each item: extracts observation, crafts a structured LLM prompt with full policy context, parses JSON response, submits action via `/step`
4. Falls back to KEEP if JSON parsing fails
5. Logs exact OpenEnv format: `[START]`, `[STEP]` (per item), `[END]` (per task)
6. Handles all 3 tasks in under 20 minutes

**System prompt:** 400+ lines covering content policies, evasion pattern recognition, escalation triggers, near-miss examples, and context utilization rules.

---

## Multiple Access Interfaces

### 1. REST HTTP API
Standard OpenEnv interface. Any language, any agent framework. Documented at `/docs`.

### 2. Gradio Web UI (`gradio_ui.py`)
Browser-based interface for manual testing or demonstration:
- Task selector + session management
- Five-action radio buttons
- Policy violated dropdown + reasoning textarea
- Confidence slider (0.0–1.0)
- Real-time observation and reward display
- Enabled with `ENABLE_WEB_INTERFACE=true`

### 3. MCP Protocol (`mcp_server.py`)
Model Context Protocol interface allowing Claude and other AI agents to interact directly. Exposes 4 tools:
- `reset_moderation_env(task_name, session_id)`
- `moderate_content(action_type, target_item_id, reasoning, confidence, session_id, policy_violated)`
- `get_moderation_state(session_id)`
- `list_moderation_tasks()`

Enabled with `MCP_MODE=true`.

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim
RUN pip install uv
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY server/ ./server/
COPY openenv.yaml .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Security: non-root user, explicit health check, pinned dependencies via `uv.lock`.

### HuggingFace Spaces

Deployed with Docker SDK. Required secrets:

| Secret | Value |
|---|---|
| `OPENAI_API_KEY` | HuggingFace token |
| `API_BASE_URL` | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace token (fallback) |
| `ENABLE_WEB_INTERFACE` | `true` / `false` |

---

## Running Locally

```bash
# 1. Clone and install
git clone https://github.com/AryanVihan/Content-Moderator
cd Content-Moderator
pip install -r requirements.txt

# 2. Start the environment server
uvicorn server.main:app --host 0.0.0.0 --port 7860 --reload

# 3. Run the LLM agent (separate terminal)
export OPENAI_API_KEY=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:7860
python inference.py

# 4. (Optional) Start web UI
export ENABLE_WEB_INTERFACE=true
uvicorn server.main:app --host 0.0.0.0 --port 7860

# 5. (Optional) Start MCP server
export MCP_MODE=true
python server/mcp_server.py
```

### Using Docker

```bash
docker build -t meta-mod-env .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=hf_your_token \
  -e ENABLE_WEB_INTERFACE=true \
  meta-mod-env
```

### Validating OpenEnv Compliance

```bash
pip install openenv
openenv validate .
# Expected: ✓ All checks passed
```

---

## OpenEnv Compliance Checklist

| Requirement | Status |
|---|---|
| `pyproject.toml` entry point: `meta-mod-env = "server.app:main"` | ✓ |
| `openenv.yaml` with observation_space, action_space, reward, tasks | ✓ |
| `uv.lock` locked dependency versions | ✓ |
| `/reset`, `/step`, `/health` endpoints with correct schemas | ✓ |
| `[START]`, `[STEP]`, `[END]` logging format in inference.py | ✓ |
| Docker deployment with `app_port: 7860` | ✓ |
| `.openenv/hub.json` Hub registration | ✓ |
| Passes `openenv validate .` | ✓ |

---

## Tech Stack & Dependencies

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.115+ |
| Data validation | Pydantic v2 |
| ASGI server | Uvicorn |
| HTTP client | httpx |
| LLM client | OpenAI SDK (compatible with HF Router, Groq, Fireworks, OpenAI) |
| Web UI | Gradio 4.0+ |
| AI agent protocol | MCP (Model Context Protocol) |
| Package manager | uv (fast pip alternative) |
| Containerization | Docker |
| Deployment | HuggingFace Spaces |
| Language | Python 3.10+ |

**No heavy ML frameworks** (PyTorch, Transformers) — pure Python logic with OpenAI-compatible API calls. The environment itself is model-agnostic; any LLM with an OpenAI-compatible endpoint can be used as the agent.

---

## Key Technical Design Decisions

**Dense rewards over sparse:** Every step returns a reward signal. Sparse (end-of-episode only) rewards slow learning and make debugging harder. Dense feedback lets agents correct course mid-episode.

**Asymmetric penalty structure:** Modeled after real trust-and-safety priority ordering. Missing a self-harm escalation (−0.25) penalizes more than a false positive removal (−0.15), which penalizes more than missing a standard violation (−0.10). This shapes agent behavior toward the correct real-world priorities.

**Deterministic data generation (seed=42):** Benchmark scores are only meaningful if every run uses identical data. Without a fixed seed, score variance reflects data randomness, not agent capability.

**Pydantic `Field(exclude=True)` for ground truth hiding:** Ground truth answers live in the same Python objects as agent-visible data, but are automatically stripped from JSON serialization. The grader reads Python objects directly; the agent only ever sees sanitized JSON. This is cleaner than maintaining two separate data models.

**`target_item_id` validation in actions:** Prevents subtle bugs where a stateful agent submits a decision for a previous item. The server checks that the submitted `target_item_id` matches the current item, rejects mismatches with `last_action_error`, and never silently applies a wrong decision.

**UUID session isolation:** Multiple agents can run concurrently with independent state. No global state, no session collision, no need for database.

**Per-task score normalization with known max rewards:** Raw cumulative rewards are normalized using task-specific constants (Task 1: 4.35, Task 2: 6.30, Task 3: 10.30) derived from the maximum achievable reward for each task. This makes scores from different-length tasks directly comparable.

---

## Project By The Numbers

| Metric | Value |
|---|---|
| Total benchmark items | 100 (20 + 30 + 50) |
| Difficulty levels | 3 (Easy / Medium / Hard) |
| Content categories | 8+ (hate speech, spam, violence, safe, satire, journalism, evasion, coordinated) |
| Evasion techniques modeled | 4 (char substitution, emoji, dogwhistles, romanized slurs) |
| API endpoints | 7 |
| Pydantic models | 8 |
| Access interfaces | 3 (REST API, Gradio UI, MCP protocol) |
| Reward components | 10+ per step |
| Score for Qwen2.5-72B (Easy) | 0.88 |
| Score for Qwen2.5-72B (Medium) | 0.62 |
| Score for Qwen2.5-72B (Hard) | 0.475 |
| Overall benchmark score | 0.658 |
| System prompt length | 400+ lines |
| Content templates | 50+ per category |
| Docker healthcheck interval | 30s |
| Max concurrent sessions | Unlimited (UUID-isolated) |
| Deployment platform | HuggingFace Spaces (live) |
| OpenEnv compliance | Full (passes `openenv validate .`) |

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | API key for LLM inference |
| `API_BASE_URL` | Yes | — | OpenAI-compatible base URL |
| `MODEL_NAME` | Yes | — | Model identifier |
| `ENV_URL` | Yes (inference) | — | Environment server URL |
| `HF_TOKEN` | No | — | HuggingFace token (fallback key) |
| `ENABLE_WEB_INTERFACE` | No | `false` | Mount Gradio UI at `/ui` |
| `MCP_MODE` | No | `false` | Start MCP server instead of HTTP |

---

## License

MIT License. See `LICENSE` for details.

---

*Built for the OpenEnv Hackathon. Simulates Meta/Facebook content moderation as a reproducible AI agent benchmark.*
