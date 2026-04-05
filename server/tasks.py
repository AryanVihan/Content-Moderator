"""Task definitions and grader logic for all 3 tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
# Grader functions
# ---------------------------------------------------------------------------

def _count_by_action(items: List[ContentItem], action: str) -> int:
    return sum(1 for i in items if i.ground_truth_action == action)


def grade_basic(
    items: List[ContentItem],
    reviewed: List[ReviewedItem],
) -> Tuple[float, Dict[str, Any]]:
    """
    score = correct_removes/total_removes * 0.35
          + correct_keeps/total_keeps    * 0.25
          + correct_escalates/total_esc  * 0.25
          + correct_labels/total_labels  * 0.15
    """
    decision_map: Dict[str, str] = {r.item_id: r.action_taken for r in reviewed}

    total_remove = _count_by_action(items, "REMOVE")
    total_keep = _count_by_action(items, "KEEP")
    total_escalate = _count_by_action(items, "ESCALATE")
    total_label = _count_by_action(items, "ADD_WARNING_LABEL")

    correct_remove = sum(
        1 for i in items
        if i.ground_truth_action == "REMOVE"
        and decision_map.get(i.item_id) == "REMOVE"
    )
    correct_keep = sum(
        1 for i in items
        if i.ground_truth_action == "KEEP"
        and decision_map.get(i.item_id) == "KEEP"
    )
    correct_escalate = sum(
        1 for i in items
        if i.ground_truth_action == "ESCALATE"
        and decision_map.get(i.item_id) == "ESCALATE"
    )
    correct_label = sum(
        1 for i in items
        if i.ground_truth_action == "ADD_WARNING_LABEL"
        and decision_map.get(i.item_id) == "ADD_WARNING_LABEL"
    )

    score = (
        (correct_remove / total_remove if total_remove else 0) * 0.35
        + (correct_keep / total_keep if total_keep else 0) * 0.25
        + (correct_escalate / total_escalate if total_escalate else 0) * 0.25
        + (correct_label / total_label if total_label else 0) * 0.15
    )

    # Floor: never exactly 0.0 (disqualification prevention), cap at 1.0
    score = float(min(max(score, 0.01), 1.0))

    breakdown = {
        "correct_removes": correct_remove,
        "total_should_remove": total_remove,
        "correct_keeps": correct_keep,
        "total_should_keep": total_keep,
        "correct_escalations": correct_escalate,
        "total_should_escalate": total_escalate,
        "correct_labels": correct_label,
        "total_should_label": total_label,
        "weighted_score": round(score, 4),
    }
    return round(score, 4), breakdown


def grade_contextual(
    items: List[ContentItem],
    reviewed: List[ReviewedItem],
) -> Tuple[float, Dict[str, Any]]:
    """
    score = (correct_decisions / total_items) * 0.50
          + context_utilization_score         * 0.30
          + false_positive_penalty            * 0.20
    """
    decision_map: Dict[str, str] = {r.item_id: r.action_taken for r in reviewed}
    total = len(items)

    correct = sum(
        1 for i in items
        if decision_map.get(i.item_id) == i.ground_truth_action
    )
    accuracy = correct / total if total else 0

    # Context utilization: % of context-dependent items decided correctly
    ctx_items = [i for i in items if i.needs_context or i.is_coordinated]
    ctx_correct = sum(
        1 for i in ctx_items
        if decision_map.get(i.item_id) == i.ground_truth_action
    )
    ctx_score = ctx_correct / len(ctx_items) if ctx_items else 1.0

    # False positive penalty: -0.05 per wrongful REMOVE of safe content
    fp_count = sum(
        1 for i in items
        if i.ground_truth_action == "KEEP"
        and decision_map.get(i.item_id) == "REMOVE"
    )
    fp_penalty = fp_count * 0.05

    score = (
        accuracy * 0.50
        + ctx_score * 0.30
        + max(0.0, (1.0 - fp_penalty)) * 0.20
    )
    # Floor: never exactly 0.0 (disqualification prevention), cap at 1.0
    score = float(min(max(score, 0.01), 1.0))

    breakdown = {
        "correct_decisions": correct,
        "total_items": total,
        "accuracy": round(accuracy, 4),
        "context_items_correct": ctx_correct,
        "context_items_total": len(ctx_items),
        "context_utilization_score": round(ctx_score, 4),
        "false_positives": fp_count,
        "false_positive_penalty": round(fp_penalty, 4),
        "weighted_score": round(score, 4),
    }
    return round(score, 4), breakdown


def grade_adversarial(
    items: List[ContentItem],
    reviewed: List[ReviewedItem],
) -> Tuple[float, Dict[str, Any]]:
    """
    score = evasion_detection_rate    * 0.30
          + urgent_escalation_catch   * 0.30
          + overall_accuracy          * 0.25
          + coordinated_flagged_rate  * 0.15
    """
    decision_map: Dict[str, str] = {r.item_id: r.action_taken for r in reviewed}
    total = len(items)

    # Evasion detection
    evasion_items = [i for i in items if i.is_evasion]
    evasion_caught = sum(
        1 for i in evasion_items
        if decision_map.get(i.item_id) == i.ground_truth_action
    )
    evasion_rate = evasion_caught / len(evasion_items) if evasion_items else 1.0

    # Urgent escalation catch
    urgent_items = [i for i in items if i.is_urgent]
    urgent_caught = sum(
        1 for i in urgent_items
        if decision_map.get(i.item_id) == "ESCALATE"
    )
    urgent_rate = urgent_caught / len(urgent_items) if urgent_items else 1.0

    # Overall accuracy
    correct = sum(
        1 for i in items
        if decision_map.get(i.item_id) == i.ground_truth_action
    )
    accuracy = correct / total if total else 0

    # Coordinated behavior flagged (REMOVE = correct action)
    coord_items = [i for i in items if i.is_coordinated]
    coord_flagged = sum(
        1 for i in coord_items
        if decision_map.get(i.item_id) == "REMOVE"
    )
    coord_rate = coord_flagged / len(coord_items) if coord_items else 1.0

    score = (
        evasion_rate * 0.30
        + urgent_rate * 0.30
        + accuracy * 0.25
        + coord_rate * 0.15
    )
    # Floor: never exactly 0.0 (disqualification prevention), cap at 1.0
    score = float(min(max(score, 0.01), 1.0))

    breakdown = {
        "evasion_items": len(evasion_items),
        "evasion_caught": evasion_caught,
        "evasion_detection_rate": round(evasion_rate, 4),
        "urgent_items": len(urgent_items),
        "urgent_caught": urgent_caught,
        "urgent_escalation_catch_rate": round(urgent_rate, 4),
        "total_items": total,
        "correct_decisions": correct,
        "overall_accuracy": round(accuracy, 4),
        "coordinated_items": len(coord_items),
        "coordinated_flagged": coord_flagged,
        "coordinated_flagged_rate": round(coord_rate, 4),
        "weighted_score": round(score, 4),
    }
    return round(score, 4), breakdown


def grade_episode(
    task_name: str,
    items: List[ContentItem],
    reviewed: List[ReviewedItem],
) -> Tuple[float, Dict[str, Any]]:
    """Dispatch to the appropriate grader."""
    if task_name == "basic_moderation":
        return grade_basic(items, reviewed)
    elif task_name == "contextual_moderation":
        return grade_contextual(items, reviewed)
    elif task_name == "adversarial_moderation":
        return grade_adversarial(items, reviewed)
    else:
        raise ValueError(f"No grader for task: {task_name}")
