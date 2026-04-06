"""Phase 1 static checks — run with: py test_phase1.py"""
import sys
import tomllib

print("=" * 55)
print("PHASE 1 — Static checks (no server needed)")
print("=" * 55)

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

# ── 1.1 Imports ──────────────────────────────────────────
print("\n[1.1] Imports ...")
try:
    from server.app import main
    from server.models import ContentItem
    from server.data_generator import (
        DataGenerator,
        EVASION_HARD_TEMPLATES,
        NEAR_MISS_TEMPLATES,
        DISGUISED_ESCALATION_TEMPLATES,
        COORDINATED_CAMPAIGN_TEMPLATES,
    )
    from server.tasks import grade_adversarial, _get_agent_action
    from server.environment import reset_episode
    print(f"  {PASS} All imports OK")
except Exception as e:
    print(f"  {FAIL} Import error: {e}")
    errors.append(str(e))
    sys.exit(1)

# ── 1.2 Data generator counts ────────────────────────────
print("\n[1.2] Data generator item counts ...")
gen = DataGenerator()

basic = gen.generate("basic_moderation")
ok = len(basic) == 20
print(f"  {'PASS' if ok else 'FAIL'} basic_moderation:      {len(basic)} items  (expected 20)")
if not ok:
    errors.append(f"basic: expected 20 got {len(basic)}")

ctx = gen.generate("contextual_moderation")
ok = len(ctx) == 30
print(f"  {'PASS' if ok else 'FAIL'} contextual_moderation: {len(ctx)} items  (expected 30)")
if not ok:
    errors.append(f"contextual: expected 30 got {len(ctx)}")

adv = gen.generate("adversarial_moderation")
ok = len(adv) == 50
print(f"  {'PASS' if ok else 'FAIL'} adversarial_moderation: {len(adv)} items  (expected 50)")
if not ok:
    errors.append(f"adversarial: expected 50 got {len(adv)}")

evasion  = [i for i in adv if i.category == "evasion"]
coord    = [i for i in adv if i.category == "coordinated"]
urgent   = [i for i in adv if i.is_urgent_escalation]
nearmiss = [i for i in adv if i.category == "near_miss"]
clear    = [i for i in adv if i.category == "clear_violation"]
border   = [i for i in adv if i.category == "borderline"]

print(f"\n  Adversarial category breakdown:")
counts = [
    ("evasion",        len(evasion),  15),
    ("coordinated",    len(coord),    10),
    ("urgent_esc",     len(urgent),    5),
    ("near_miss",      len(nearmiss),  8),
    ("clear_violation",len(clear),     7),
    ("borderline",     len(border),    5),
]
for name, got, expected in counts:
    ok = got == expected
    print(f"    {'PASS' if ok else 'FAIL'} {name:<18} {got:>2}  (expected {expected})")
    if not ok:
        errors.append(f"{name}: expected {expected} got {got}")

no_cat = [i for i in adv if i.category is None]
ok = len(no_cat) == 0
print(f"  {'PASS' if ok else 'FAIL'} No items missing category ({len(no_cat)} missing)")
if not ok:
    errors.append(f"{len(no_cat)} items missing category field")

# ── 1.3 Grader logic ─────────────────────────────────────
print("\n[1.3] Grader logic ...")
from server.models import ReviewedItem

# Empty agent
score, bd = grade_adversarial(adv, [])
ok = score == 0.01
print(f"  {'PASS' if ok else 'FAIL'} Empty agent score: {score}  (expected 0.01)")
if not ok:
    errors.append(f"empty agent: expected 0.01 got {score}")

# Agent catching only the 5 urgent escalations
urgent_only = [
    ReviewedItem(
        item_id=item.item_id,
        action_taken="ESCALATE",
        policy_cited="SELF_HARM",
        reasoning="test",
        confidence=0.9,
        reward_received=0.0,
    )
    for item in adv
    if item.is_urgent_escalation
]
score2, bd2 = grade_adversarial(adv, urgent_only)
ok = bd2["urgent_rate"] == 1.0
print(f"  {'PASS' if ok else 'FAIL'} Urgent-only agent: urgent_rate={bd2['urgent_rate']}  score={score2}  (expected urgent_rate=1.0)")
if not ok:
    errors.append(f"urgent_rate: expected 1.0 got {bd2['urgent_rate']}")

# Perfect agent on all items
perfect = [
    ReviewedItem(
        item_id=item.item_id,
        action_taken=item.ground_truth_action,
        policy_cited=item.ground_truth_policy,
        reasoning="test",
        confidence=0.95,
        reward_received=0.0,
    )
    for item in adv
]
score3, bd3 = grade_adversarial(adv, perfect)
ok = 0.35 <= score3 <= 0.80
print(f"  {'PASS' if ok else 'FAIL'} Perfect agent: raw_sum={bd3['raw_weighted_sum']}  final={score3}  (expected 0.35-0.80)")
if not ok:
    errors.append(f"perfect agent score {score3} not in 0.35-0.80")

# All-REMOVE agent (should score low — triggers near_miss false positives + misses urgents)
all_remove = [
    ReviewedItem(
        item_id=item.item_id,
        action_taken="REMOVE",
        policy_cited=None,
        reasoning="test",
        confidence=0.5,
        reward_received=0.0,
    )
    for item in adv
]
score4, bd4 = grade_adversarial(adv, all_remove)
ok = score4 < 0.55
print(f"  {'PASS' if ok else 'FAIL'} All-REMOVE agent: score={score4}  fp_component={bd4['fp_component']}  urgent_component={bd4['urgent_component']}  (should be <0.55)")
if not ok:
    errors.append(f"all-REMOVE agent scored too high: {score4}")

# ── 1.4 openenv.yaml entry_point ─────────────────────────
print("\n[1.4] openenv.yaml entry_point ...")
try:
    yaml_text = open("openenv.yaml").read()
    ok = 'entry_point: "server.app:main"' in yaml_text
    print(f"  {'PASS' if ok else 'FAIL'} entry_point: {'server.app:main' if ok else 'WRONG — check openenv.yaml'}")
    if not ok:
        errors.append("openenv.yaml entry_point is not server.app:main")
except FileNotFoundError:
    print(f"  {FAIL} openenv.yaml not found")
    errors.append("openenv.yaml not found")

# ── 1.5 pyproject.toml ───────────────────────────────────
print("\n[1.5] pyproject.toml ...")
try:
    with open("pyproject.toml", "rb") as f:
        toml = tomllib.load(f)
    backend = toml["build-system"]["build-backend"]
    script  = toml["project"]["scripts"].get("meta-mod-env", "MISSING")
    ok_b = backend == "hatchling.build"
    ok_s = script == "server.app:main"
    print(f"  {'PASS' if ok_b else 'FAIL'} build-backend: {backend}  (expected hatchling.build)")
    print(f"  {'PASS' if ok_s else 'FAIL'} script: meta-mod-env = {script}  (expected server.app:main)")
    if not ok_b:
        errors.append(f"build-backend wrong: {backend}")
    if not ok_s:
        errors.append(f"script wrong: {script}")
except Exception as e:
    print(f"  {FAIL} {e}")
    errors.append(str(e))

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
if errors:
    print(f"RESULT: {len(errors)} FAILURE(s)")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 55)
