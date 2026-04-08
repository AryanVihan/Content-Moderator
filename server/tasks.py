"""Task definitions and grader logic for all 3 tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

from server.models import ContentItem, ReviewedItem


@dataclass
class TaskConfig:
    name: str
    description: str
    difficulty: str
    queue_size: int
    max_steps: int
    expected_score_min: float
    expected_score_max: float
    goal: str


TASKS: Dict[str, TaskConfig] = {
    "basic_moderation": TaskConfig(
        name="basic_moderation",
        description=(
            "20-item queue of clear-cut violations, safe content, borderline items "
            "needing warning labels, and urgent escalations. No ambiguity — "
            "straightforward moderation decisions."
        ),
        difficulty="EASY",
        queue_size=20,
        max_steps=25,
        expected_score_min=0.80,
        expected_score_max=0.95,
        goal=(
            "Review each flagged content item in the queue and make the correct "
            "moderation decision: REMOVE clear violations, KEEP safe content, "
            "ADD_WARNING_LABEL to borderline items, and ESCALATE urgent cases "
            "(self-harm, credible threats). Use Meta Community Standards."
        ),
    ),
    "contextual_moderation": TaskConfig(
        name="contextual_moderation",
        description=(
            "30-item queue where context determines the correct decision. "
            "Satire accounts, journalist content, reclaimed language, misleading "
            "framing, and coordinated inauthentic behavior signals present."
        ),
        difficulty="MEDIUM",
        queue_size=30,
        max_steps=40,
        expected_score_min=0.50,
        expected_score_max=0.70,
        goal=(
            "Review each flagged item carefully using all available context clues: "
            "account type, verification status, user history, report patterns, and "
            "content framing. The SAME surface text can be fine or a violation "
            "depending on who posted it and why. Watch for coordinated inauthentic "
            "behavior across multiple items."
        ),
    ),
    "adversarial_moderation": TaskConfig(
        name="adversarial_moderation",
        description=(
            "50-item queue designed to trick the agent. Policy circumvention "
            "(misspellings, emoji substitution, Romanized cross-language content), "
            "viral misinformation, political speech edge cases, coordinated "
            "brigading campaigns, and 5 urgent escalations buried in the queue."
        ),
        difficulty="HARD",
        queue_size=50,
        max_steps=60,
        expected_score_min=0.35,
        expected_score_max=0.55,
        goal=(
            "Review each item and detect policy circumvention attempts, coordinated "
            "brigading, and viral misinformation despite high removal costs. "
            "CRITICAL: identify the 5 urgent escalation cases buried in the queue — "
            "missing them incurs the highest penalty. Political speech from public "
            "figures may look like incitement but be legally protected — use context."
        ),
    ),
}


def get_task(task_name: str) -> TaskConfig:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")
    return TASKS[task_name]


# ---------------------------------------------------------------------------
# Duck-type helpers (accept both Pydantic objects and plain dicts)
# ---------------------------------------------------------------------------

def _gt_attr(item: Any, attr: str) -> Any:
    """Get a field from a ground-truth item that may be a dict or a Pydantic model."""
    if isinstance(item, dict):
        return item.get(attr)
    return getattr(item, attr, None)


def _build_decision_map(actions: List[Any]) -> Dict[str, str]:
    """Build {item_id -> action_type} from a list of action dicts or ReviewedItem objects."""
    result: Dict[str, str] = {}
    for a in actions:
        if isinstance(a, dict):
            item_id = a.get("target_item_id") or a.get("item_id")
            action = a.get("action_type") or a.get("action_taken")
        else:
            item_id = getattr(a, "item_id", None) or getattr(a, "target_item_id", None)
            action = getattr(a, "action_taken", None) or getattr(a, "action_type", None)
        if item_id and action:
            result[item_id] = action
    return result


# ---------------------------------------------------------------------------
# Internal graders — return (score, breakdown) tuples, used by grade_episode
# ---------------------------------------------------------------------------

def _count_by_action(items: List[ContentItem], action: str) -> int:
    return sum(1 for i in items if i.ground_truth_action == action)


def _grade_basic_full(
    items: List[ContentItem],
    reviewed: List[ReviewedItem],
) -> Tuple[float, Dict[str, Any]]:
    decision_map: Dict[str, str] = {r.item_id: r.action_taken for r in reviewed}

    total_remove = _count_by_action(items, "REMOVE")
    total_keep = _count_by_action(items, "KEEP")
    total_escalate = _count_by_action(items, "ESCALATE")
    total_label = _count_by_action(items, "ADD_WARNING_LABEL")

    correct_remove = sum(1 for i in items if i.ground_truth_action == "REMOVE" and decision_map.get(i.item_id) == "REMOVE")
    correct_keep = sum(1 for i in items if i.ground_truth_action == "KEEP" and decision_map.get(i.item_id) == "KEEP")
    correct_escalate = sum(1 for i in items if i.ground_truth_action == "ESCALATE" and decision_map.get(i.item_id) == "ESCALATE")
    correct_label = sum(1 for i in items if i.ground_truth_action == "ADD_WARNING_LABEL" and decision_map.get(i.item_id) == "ADD_WARNING_LABEL")

    score = (
        (correct_remove / total_remove if total_remove else 0) * 0.35
        + (correct_keep / total_keep if total_keep else 0) * 0.25
        + (correct_escalate / total_escalate if total_escalate else 0) * 0.25
        + (correct_label / total_label if total_label else 0) * 0.15
    )
    score = float(min(max(score, 0.01), 1.0))
    breakdown = {
        "correct_removes": correct_remove, "total_should_remove": total_remove,
        "correct_keeps": correct_keep, "total_should_keep": total_keep,
        "correct_escalations": correct_escalate, "total_should_escalate": total_escalate,
        "correct_labels": correct_label, "total_should_label": total_label,
        "weighted_score": round(score, 4),
    }
    return round(score, 4), breakdown


def _grade_contextual_full(
    items: List[ContentItem],
    reviewed: List[ReviewedItem],
) -> Tuple[float, Dict[str, Any]]:
    decision_map: Dict[str, str] = {r.item_id: r.action_taken for r in reviewed}
    total = len(items)
    correct = sum(1 for i in items if decision_map.get(i.item_id) == i.ground_truth_action)
    accuracy = correct / total if total else 0
    ctx_items = [i for i in items if i.needs_context or i.is_coordinated]
    ctx_correct = sum(1 for i in ctx_items if decision_map.get(i.item_id) == i.ground_truth_action)
    ctx_score = ctx_correct / len(ctx_items) if ctx_items else 1.0
    fp_count = sum(1 for i in items if i.ground_truth_action == "KEEP" and decision_map.get(i.item_id) == "REMOVE")
    fp_penalty = fp_count * 0.05
    score = accuracy * 0.50 + ctx_score * 0.30 + max(0.0, (1.0 - fp_penalty)) * 0.20
    score = float(min(max(score, 0.01), 1.0))
    breakdown = {
        "correct_decisions": correct, "total_items": total, "accuracy": round(accuracy, 4),
        "context_items_correct": ctx_correct, "context_items_total": len(ctx_items),
        "context_utilization_score": round(ctx_score, 4),
        "false_positives": fp_count, "false_positive_penalty": round(fp_penalty, 4),
        "weighted_score": round(score, 4),
    }
    return round(score, 4), breakdown


def _get_agent_action_from_map(decision_map: Dict[str, str], item_id: str) -> Optional[str]:
    return decision_map.get(item_id)


def _grade_adversarial_full(
    items: List[ContentItem],
    reviewed: List[ReviewedItem],
) -> Tuple[float, Dict[str, Any]]:
    """
    Strict multi-component grader designed to produce scores in 0.35–0.55 range.

    Component 1 — Evasion detection          (30% weight, partial credit only if >50% correct)
    Component 2 — Urgent escalation catch    (30% weight, sharp binary penalty curve)
    Component 3 — False positive rate        (20% weight, sharp penalty for near-miss removals)
    Component 4 — Coordinated campaign       (15% weight)
    Component 5 — Other accuracy             ( 5% weight)

    A difficulty_multiplier of 0.65 is applied to the weighted sum so that
    even a strong component score lands in the target range.
    """
    if not reviewed:
        return 0.01, {"error": "no_actions", "weighted_score": 0.01}

    total_score = 0.0

    # --- Component 1: Evasion detection (30%) ---
    decision_map: Dict[str, str] = {r.item_id: r.action_taken for r in reviewed}

    evasion_items = [i for i in items if i.category == "evasion"]
    if evasion_items:
        evasion_correct = sum(
            1 for item in evasion_items
            if _get_agent_action_from_map(decision_map, item.item_id) == item.ground_truth_action
        )
        evasion_rate = evasion_correct / len(evasion_items)
        evasion_score = evasion_rate if evasion_rate > 0.5 else evasion_rate * 0.6
        total_score += evasion_score * 0.30
    else:
        evasion_correct, evasion_rate, evasion_score = 0, 0.0, 0.0

    urgent_items = [i for i in items if i.is_urgent_escalation]
    if urgent_items:
        urgent_caught = sum(
            1 for item in urgent_items
            if _get_agent_action_from_map(decision_map, item.item_id) == "ESCALATE"
        )
        urgent_rate = urgent_caught / len(urgent_items)
        if urgent_rate >= 1.0:
            urgent_score = 1.0
        elif urgent_rate >= 0.8:
            urgent_score = 0.65
        elif urgent_rate >= 0.6:
            urgent_score = 0.35
        elif urgent_rate >= 0.4:
            urgent_score = 0.20
        elif urgent_rate > 0:
            urgent_score = 0.10
        else:
            urgent_score = 0.0
        total_score += urgent_score * 0.30
    else:
        urgent_caught, urgent_rate, urgent_score = 0, 0.0, 0.0

    near_miss_items = [i for i in items if i.category == "near_miss"]
    if near_miss_items:
        false_positives = sum(
            1 for item in near_miss_items
            if _get_agent_action_from_map(decision_map, item.item_id) == "REMOVE"
        )
        fp_rate = false_positives / len(near_miss_items)
        fp_score = max(0.0, 1.0 - (fp_rate * 2.0))
        total_score += fp_score * 0.20
    else:
        false_positives, fp_rate, fp_score = 0, 0.0, 1.0

    coord_items = [i for i in items if i.category == "coordinated"]
    if coord_items:
        coord_correct = sum(
            1 for item in coord_items
            if _get_agent_action_from_map(decision_map, item.item_id) == item.ground_truth_action
        )
        coord_rate = coord_correct / len(coord_items)
        total_score += coord_rate * 0.15
    else:
        coord_correct, coord_rate = 0, 0.0

    other_items = [
        i for i in items
        if i.category not in ("evasion", "near_miss", "coordinated")
        and not i.is_urgent_escalation
    ]
    if other_items:
        other_correct = sum(
            1 for item in other_items
            if _get_agent_action_from_map(decision_map, item.item_id) == item.ground_truth_action
        )
        other_rate = other_correct / len(other_items)
        total_score += other_rate * 0.05
    else:
        other_correct, other_rate = 0, 0.0

    difficulty_multiplier = 0.65
    final_score = float(min(max(total_score * difficulty_multiplier, 0.01), 1.0))

    breakdown = {
        "evasion_items": len(evasion_items), "evasion_correct": evasion_correct,
        "evasion_rate": round(evasion_rate, 4), "evasion_component": round(evasion_score * 0.30, 4),
        "urgent_items": len(urgent_items), "urgent_caught": urgent_caught,
        "urgent_rate": round(urgent_rate, 4), "urgent_component": round(urgent_score * 0.30, 4),
        "near_miss_items": len(near_miss_items), "false_positives": false_positives,
        "fp_rate": round(fp_rate, 4), "fp_component": round(fp_score * 0.20, 4),
        "coordinated_items": len(coord_items), "coordinated_correct": coord_correct,
        "coordinated_rate": round(coord_rate, 4), "coordinated_component": round(coord_rate * 0.15, 4),
        "other_items": len(other_items), "other_correct": other_correct,
        "other_rate": round(other_rate, 4), "other_component": round(other_rate * 0.05, 4),
        "raw_weighted_sum": round(total_score, 4),
        "difficulty_multiplier": difficulty_multiplier,
        "weighted_score": final_score,
    }
    return final_score, breakdown


# ---------------------------------------------------------------------------
# Public grade functions — check-script interface: (actions, ground_truth) -> float
# actions: list of dicts {'target_item_id', 'action_type', ...} or ReviewedItem objects
# ground_truth: list of dicts {'item_id', 'ground_truth_action', ...} or ContentItem objects
# ---------------------------------------------------------------------------

def grade_basic(actions: List[Any], ground_truth: List[Any]) -> float:
    """
    score = correct_removes/total_removes * 0.35
          + correct_keeps/total_keeps    * 0.25
          + correct_escalates/total_esc  * 0.25
          + correct_labels/total_labels  * 0.15
    """
    decision_map = _build_decision_map(actions)

    def _count(gt_action: str) -> int:
        return sum(1 for i in ground_truth if _gt_attr(i, "ground_truth_action") == gt_action)

    def _correct(gt_action: str) -> int:
        return sum(
            1 for i in ground_truth
            if _gt_attr(i, "ground_truth_action") == gt_action
            and decision_map.get(_gt_attr(i, "item_id")) == gt_action
        )

    total_remove, total_keep = _count("REMOVE"), _count("KEEP")
    total_escalate, total_label = _count("ESCALATE"), _count("ADD_WARNING_LABEL")
    score = (
        (_correct("REMOVE") / total_remove if total_remove else 0) * 0.35
        + (_correct("KEEP") / total_keep if total_keep else 0) * 0.25
        + (_correct("ESCALATE") / total_escalate if total_escalate else 0) * 0.25
        + (_correct("ADD_WARNING_LABEL") / total_label if total_label else 0) * 0.15
    )
    return float(round(min(max(score, 0.01), 1.0), 4))


def grade_contextual(actions: List[Any], ground_truth: List[Any]) -> float:
    """
    score = (correct_decisions / total_items) * 0.50
          + context_utilization_score         * 0.30
          + false_positive_penalty            * 0.20
    """
    decision_map = _build_decision_map(actions)
    total = len(ground_truth)
    correct = sum(
        1 for i in ground_truth
        if decision_map.get(_gt_attr(i, "item_id")) == _gt_attr(i, "ground_truth_action")
    )
    accuracy = correct / total if total else 0
    ctx_items = [i for i in ground_truth if _gt_attr(i, "needs_context") or _gt_attr(i, "is_coordinated")]
    ctx_correct = sum(
        1 for i in ctx_items
        if decision_map.get(_gt_attr(i, "item_id")) == _gt_attr(i, "ground_truth_action")
    )
    ctx_score = ctx_correct / len(ctx_items) if ctx_items else 1.0
    fp_count = sum(
        1 for i in ground_truth
        if _gt_attr(i, "ground_truth_action") == "KEEP"
        and decision_map.get(_gt_attr(i, "item_id")) == "REMOVE"
    )
    fp_penalty = fp_count * 0.05
    score = accuracy * 0.50 + ctx_score * 0.30 + max(0.0, (1.0 - fp_penalty)) * 0.20
    return float(round(min(max(score, 0.01), 1.0), 4))


def grade_adversarial(actions: List[Any], ground_truth: List[Any]) -> float:
    """
    Multi-component grader designed to produce scores in 0.35–0.55 range.
    Component 1 — Evasion detection          (30%)
    Component 2 — Urgent escalation catch    (30%)
    Component 3 — False positive rate        (20%)
    Component 4 — Coordinated campaign       (15%)
    Component 5 — Other accuracy             ( 5%)
    """
    if not actions:
        return 0.01

    decision_map = _build_decision_map(actions)
    total_score = 0.0

    evasion_items = [i for i in ground_truth if _gt_attr(i, "category") == "evasion"]
    if evasion_items:
        evasion_correct = sum(
            1 for item in evasion_items
            if decision_map.get(_gt_attr(item, "item_id")) == _gt_attr(item, "ground_truth_action")
        )
        evasion_rate = evasion_correct / len(evasion_items)
        evasion_score = evasion_rate if evasion_rate > 0.5 else evasion_rate * 0.6
        total_score += evasion_score * 0.30
    else:
        evasion_rate, evasion_score = 0.0, 0.0

    urgent_items = [i for i in ground_truth if _gt_attr(i, "is_urgent_escalation")]
    if urgent_items:
        urgent_caught = sum(
            1 for item in urgent_items
            if decision_map.get(_gt_attr(item, "item_id")) == "ESCALATE"
        )
        urgent_rate = urgent_caught / len(urgent_items)
        if urgent_rate >= 1.0:
            urgent_score = 1.0
        elif urgent_rate >= 0.8:
            urgent_score = 0.65
        elif urgent_rate >= 0.6:
            urgent_score = 0.35
        elif urgent_rate >= 0.4:
            urgent_score = 0.20
        elif urgent_rate > 0:
            urgent_score = 0.10
        else:
            urgent_score = 0.0
        total_score += urgent_score * 0.30
    else:
        urgent_rate, urgent_score = 0.0, 0.0

    near_miss_items = [i for i in ground_truth if _gt_attr(i, "category") == "near_miss"]
    if near_miss_items:
        false_positives = sum(
            1 for item in near_miss_items
            if decision_map.get(_gt_attr(item, "item_id")) == "REMOVE"
        )
        fp_rate = false_positives / len(near_miss_items)
        fp_score = max(0.0, 1.0 - (fp_rate * 2.0))
        total_score += fp_score * 0.20
    else:
        fp_score = 1.0
        total_score += fp_score * 0.20

    coord_items = [i for i in ground_truth if _gt_attr(i, "category") == "coordinated"]
    if coord_items:
        coord_correct = sum(
            1 for item in coord_items
            if decision_map.get(_gt_attr(item, "item_id")) == _gt_attr(item, "ground_truth_action")
        )
        total_score += (coord_correct / len(coord_items)) * 0.15

    other_items = [
        i for i in ground_truth
        if _gt_attr(i, "category") not in ("evasion", "near_miss", "coordinated")
        and not _gt_attr(i, "is_urgent_escalation")
    ]
    if other_items:
        other_correct = sum(
            1 for item in other_items
            if decision_map.get(_gt_attr(item, "item_id")) == _gt_attr(item, "ground_truth_action")
        )
        total_score += (other_correct / len(other_items)) * 0.05

    difficulty_multiplier = 0.65
    return float(round(min(max(total_score * difficulty_multiplier, 0.01), 1.0), 4))


def grade_episode(
    task_name: str,
    items: List[ContentItem],
    reviewed: List[ReviewedItem],
) -> Tuple[float, Dict[str, Any]]:
    """Dispatch to the appropriate internal grader (returns tuple for environment use)."""
    if task_name == "basic_moderation":
        return _grade_basic_full(items, reviewed)
    elif task_name == "contextual_moderation":
        return _grade_contextual_full(items, reviewed)
    elif task_name == "adversarial_moderation":
        return _grade_adversarial_full(items, reviewed)
    else:
        raise ValueError(f"No grader for task: {task_name}")
