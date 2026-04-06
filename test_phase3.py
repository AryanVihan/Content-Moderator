"""
Phase 3 — Live API tests against a running server.
Usage:  py test_phase3.py [base_url]
Default base_url: http://localhost:7860

Start the server first:
  py -m uvicorn server.main:app --host 0.0.0.0 --port 7860
"""
import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
print(f"Testing against: {BASE}\n")

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

client = httpx.Client(timeout=30)


def check(label: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    print(f"  {tag} {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        errors.append(label)


# ── 3.1 Health ───────────────────────────────────────────
print("[3.1] Health check ...")
try:
    r = client.get(f"{BASE}/health")
    check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    check('body == {"status":"ok"}', r.json() == {"status": "ok"}, r.text)
except Exception as e:
    print(f"  {FAIL} Cannot reach server: {e}")
    print("  → Start the server first: py -m uvicorn server.main:app --host 0.0.0.0 --port 7860")
    sys.exit(1)

# ── 3.2 Task list ────────────────────────────────────────
print("\n[3.2] Task list ...")
r = client.get(f"{BASE}/tasks")
check("HTTP 200", r.status_code == 200)
tasks = r.json()
names = [t["name"] for t in tasks]
check("3 tasks returned", len(tasks) == 3, str(names))
check("basic_moderation present", "basic_moderation" in names)
check("contextual_moderation present", "contextual_moderation" in names)
check("adversarial_moderation present", "adversarial_moderation" in names)

adv_task = next((t for t in tasks if t["name"] == "adversarial_moderation"), {})
check("adversarial expected_score_min=0.35", adv_task.get("expected_score_min") == 0.35, str(adv_task.get("expected_score_min")))
check("adversarial expected_score_max=0.55", adv_task.get("expected_score_max") == 0.55, str(adv_task.get("expected_score_max")))
check("adversarial queue_size=50", adv_task.get("queue_size") == 50, str(adv_task.get("queue_size")))

# ── 3.3 Reset all three tasks ────────────────────────────
print("\n[3.3] Reset all three tasks ...")
sessions = {}
for task_name, expected_total in [
    ("basic_moderation", 20),
    ("contextual_moderation", 30),
    ("adversarial_moderation", 50),
]:
    sid = f"phase3_{task_name}"
    r = client.post(f"{BASE}/reset", json={"task_name": task_name, "session_id": sid})
    check(f"reset {task_name} HTTP 200", r.status_code == 200, f"got {r.status_code}")
    body = r.json()
    obs = body.get("observation") or {}
    check(f"{task_name} not done", body.get("done") is False)
    check(f"{task_name} has current_item", bool(obs.get("current_item")))
    check(f"{task_name} queue_total={expected_total}", obs.get("queue_total") == expected_total, str(obs.get("queue_total")))
    check(f"{task_name} queue_position=1", obs.get("queue_position") == 1, str(obs.get("queue_position")))
    sessions[task_name] = {"sid": sid, "obs": obs}
    print(f"    item_id: {obs.get('current_item', {}).get('item_id')}  |  content preview: {str(obs.get('current_item', {}).get('content_text', ''))[:60]}...")

# ── 3.4 Submit a step ────────────────────────────────────
print("\n[3.4] Submit a step action (basic_moderation) ...")
sid   = sessions["basic_moderation"]["sid"]
obs   = sessions["basic_moderation"]["obs"]
item  = obs.get("current_item", {})
iid   = item.get("item_id", "unknown")

r = client.post(f"{BASE}/step", json={
    "session_id": sid,
    "action": {
        "action_type": "REMOVE",
        "target_item_id": iid,
        "policy_violated": "HATE_SPEECH",
        "reasoning": "Phase 3 test step",
        "confidence": 0.9,
    },
})
check("step HTTP 200", r.status_code == 200, f"got {r.status_code}")
body = r.json()
reward = body.get("reward", {})
check("reward.value present", "value" in reward, str(reward))
check("reward.value in [-1,1]", -1.0 <= reward.get("value", 999) <= 1.0, str(reward.get("value")))
check("reward.cumulative present", "cumulative" in reward)
print(f"    reward.value={reward.get('value')}  cumulative={reward.get('cumulative')}")

# ── 3.5 State endpoint ───────────────────────────────────
print("\n[3.5] State endpoint ...")
r = client.get(f"{BASE}/state", params={"session_id": sid})
check("state HTTP 200", r.status_code == 200, f"got {r.status_code}")
state = r.json()
check("step=1 after one action", state.get("step") == 1, str(state.get("step")))
check("reviewed list has 1 entry", len(state.get("reviewed", [])) == 1)
check("current_index advanced", state.get("current_index") == 1, str(state.get("current_index")))

# ── 3.6 ESCALATE action on adversarial ───────────────────
print("\n[3.6] ESCALATE action on adversarial_moderation ...")
sid = sessions["adversarial_moderation"]["sid"]
obs = sessions["adversarial_moderation"]["obs"]
iid = obs.get("current_item", {}).get("item_id", "unknown")

r = client.post(f"{BASE}/step", json={
    "session_id": sid,
    "action": {
        "action_type": "ESCALATE",
        "target_item_id": iid,
        "policy_violated": "SELF_HARM",
        "reasoning": "Detected self-harm signal",
        "confidence": 0.85,
    },
})
check("escalate HTTP 200", r.status_code == 200, f"got {r.status_code}")
body = r.json()
print(f"    reward.value={body.get('reward', {}).get('value')}  done={body.get('done')}")

# ── 3.7 Invalid task name ────────────────────────────────
print("\n[3.7] Invalid task name error handling ...")
r = client.post(f"{BASE}/reset", json={"task_name": "does_not_exist", "session_id": "err_test"})
check("HTTP 200 (not 500)", r.status_code == 200, f"got {r.status_code}")
body = r.json()
check("done=True on invalid task", body.get("done") is True, str(body.get("done")))
check("error message present", bool(body.get("info", {}).get("error")), str(body.get("info")))
print(f"    error: {body.get('info', {}).get('error')}")

# ── 3.8 Sessions endpoint ────────────────────────────────
print("\n[3.8] Sessions endpoint ...")
r = client.get(f"{BASE}/sessions")
check("sessions HTTP 200", r.status_code == 200)
check("sessions key present", "sessions" in r.json())
print(f"    active sessions: {r.json().get('sessions')}")

# ── Summary ───────────────────────────────────────────────
client.close()
print("\n" + "=" * 55)
if errors:
    print(f"RESULT: {len(errors)} FAILURE(s)")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("RESULT: ALL API CHECKS PASSED")
print("=" * 55)
