"""Per-step and end-of-episode reward logic."""
from __future__ import annotations

from typing import Dict, Tuple

from server.models import Action, ContentItem

# ---------------------------------------------------------------------------
# Per-step reward constants
# ---------------------------------------------------------------------------
R_CORRECT_REMOVE = 0.20
R_CORRECT_KEEP = 0.15
R_CORRECT_ESCALATE = 0.20
R_CORRECT_LABEL = 0.15
R_HIGH_CONFIDENCE_BONUS = 0.05   # confidence > 0.8 on correct decision
R_FALSE_POSITIVE_REMOVE = -0.15  # wrongly removed safe content
R_MISSED_ESCALATION = -0.25      # kept/removed instead of escalating urgent item
R_KEEP_ON_VIOLATION = -0.10      # kept clear violation
R_LOW_CONFIDENCE_PENALTY = -0.05 # confidence < 0.4 on obvious decision
R_WRONG_POLICY = -0.20           # REMOVE with wrong policy_violated field

VALID_POLICIES = {
    "HATE_SPEECH",
    "VIOLENCE",
    "MISINFORMATION",
    "SPAM",
    "SELF_HARM",
    "HARASSMENT",
    "SEXUAL_CONTENT",
}


def compute_step_reward(
    action: Action,
    item: ContentItem,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (clamped_reward, breakdown_dict).
    Clamps result to [-1.0, 1.0].
    """
    breakdown: Dict[str, float] = {}
    total = 0.0

    gt = item.ground_truth_action
    gt_policy = item.ground_truth_policy
    act = action.action_type
    conf = action.confidence
    is_urgent = item.is_urgent

    # --- Missed escalation (most severe) ---
    if is_urgent and act != "ESCALATE":
        breakdown["missed_escalation"] = R_MISSED_ESCALATION
        total += R_MISSED_ESCALATION

    # --- Correct decisions ---
    elif act == "REMOVE" and gt == "REMOVE":
        breakdown["correct_remove"] = R_CORRECT_REMOVE
        total += R_CORRECT_REMOVE
        # Verify policy
        if action.policy_violated:
            cited = (action.policy_violated or "").upper().strip()
            if cited not in VALID_POLICIES:
                breakdown["wrong_policy"] = R_WRONG_POLICY
                total += R_WRONG_POLICY
            elif gt_policy and cited != gt_policy.upper():
                breakdown["wrong_policy"] = R_WRONG_POLICY
                total += R_WRONG_POLICY

    elif act == "KEEP" and gt == "KEEP":
        breakdown["correct_keep"] = R_CORRECT_KEEP
        total += R_CORRECT_KEEP

    elif act == "ESCALATE" and gt == "ESCALATE":
        breakdown["correct_escalate"] = R_CORRECT_ESCALATE
        total += R_CORRECT_ESCALATE

    elif act == "ADD_WARNING_LABEL" and gt == "ADD_WARNING_LABEL":
        breakdown["correct_label"] = R_CORRECT_LABEL
        total += R_CORRECT_LABEL

    # --- Wrong decisions ---
    elif act == "REMOVE" and gt == "KEEP":
        breakdown["false_positive_remove"] = R_FALSE_POSITIVE_REMOVE
        total += R_FALSE_POSITIVE_REMOVE

    elif act == "KEEP" and gt in ("REMOVE",):
        breakdown["keep_on_violation"] = R_KEEP_ON_VIOLATION
        total += R_KEEP_ON_VIOLATION

    # REQUEST_CONTEXT is treated neutrally (0 reward)

    # --- Confidence modifiers ---
    is_correct = (
        (act == gt) or
        (act == "ESCALATE" and is_urgent)
    )
    is_obvious = gt in ("REMOVE", "KEEP") and not item.needs_context

    if is_correct and conf > 0.8:
        breakdown["high_confidence_bonus"] = R_HIGH_CONFIDENCE_BONUS
        total += R_HIGH_CONFIDENCE_BONUS

    if is_obvious and conf < 0.4:
        breakdown["low_confidence_penalty"] = R_LOW_CONFIDENCE_PENALTY
        total += R_LOW_CONFIDENCE_PENALTY

    clamped = max(-1.0, min(1.0, total))
    return clamped, breakdown


def compute_episode_bonus(
    reviewed_items: list,
    total_items: int,
    steps_used: int,
    max_steps: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute end-of-episode bonuses.
    reviewed_items: list of (ContentItem, action_type) tuples.
    Returns (bonus, breakdown).
    """
    breakdown: Dict[str, float] = {}
    bonus = 0.0

    if total_items == 0:
        return 0.0, breakdown

    correct = sum(
        1 for item, act in reviewed_items
        if act == item.ground_truth_action
    )
    accuracy = correct / total_items

    urgent_items = [(item, act) for item, act in reviewed_items if item.is_urgent]
    all_urgent_caught = all(act == "ESCALATE" for item, act in urgent_items) if urgent_items else True

    if accuracy > 0.85:
        breakdown["accuracy_bonus"] = 0.15
        bonus += 0.15

    if all_urgent_caught and urgent_items:
        breakdown["all_escalations_caught"] = 0.10
        bonus += 0.10

    if steps_used <= max_steps:
        breakdown["within_budget"] = 0.05
        bonus += 0.05

    return bonus, breakdown


def normalize_score(
    raw_reward: float,
    min_possible: float,
    max_possible: float,
) -> float:
    """Normalize raw cumulative reward to [0.0, 1.0]."""
    if max_possible == min_possible:
        return 0.0
    score = (raw_reward - min_possible) / (max_possible - min_possible)
    return max(0.0, min(1.0, score))


class RewardCalculator:
    """Convenience wrapper around module-level reward functions."""

    def compute_step_reward(self, action: Action, item: "ContentItem") -> Tuple[float, Dict[str, float]]:
        return compute_step_reward(action, item)

    def compute_episode_bonus(self, reviewed_items: list, total_items: int, steps_used: int, max_steps: int) -> Tuple[float, Dict[str, float]]:
        return compute_episode_bonus(reviewed_items, total_items, steps_used, max_steps)

    def normalize_score(self, raw_reward: float, min_possible: float, max_possible: float) -> float:
        return normalize_score(raw_reward, min_possible, max_possible)
