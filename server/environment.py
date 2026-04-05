"""Core OpenEnv environment logic for MetaModEnv."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from server.data_generator import DataGenerator
from server.models import (
    Action,
    AgentVisibleContentItem,
    ContentItem,
    Observation,
    Reward,
    ReviewedItem,
    StepResult,
)
from server.rewards import compute_episode_bonus, compute_step_reward, normalize_score
from server.tasks import TaskConfig, get_task, grade_episode

_generator = DataGenerator()


class Episode:
    """Single episode state — one task run for one session."""

    def __init__(self, task_name: str, session_id: str):
        self.session_id = session_id
        self.task_config: TaskConfig = get_task(task_name)
        self.task_name = task_name

        # Load queue
        self.queue: List[ContentItem] = _generator.generate(task_name)
        self.current_index: int = 0
        self.reviewed: List[ReviewedItem] = []
        self.reviewed_items_with_actions: List[Tuple[ContentItem, str]] = []

        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.last_action_error: Optional[str] = None

        # Theoretical bounds for normalization
        # Best case: every step correct + high conf bonus + end bonuses
        self._max_possible = (
            len(self.queue) * (0.20 + 0.05) + 0.30
        )
        # Worst case: every step missed escalation
        self._min_possible = len(self.queue) * (-0.25)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def current_item(self) -> Optional[ContentItem]:
        if self.current_index < len(self.queue):
            return self.queue[self.current_index]
        return None

    def _strip_item(self, item: ContentItem) -> AgentVisibleContentItem:
        return AgentVisibleContentItem(
            item_id=item.item_id,
            platform=item.platform,
            content_text=item.content_text,
            user_history=item.user_history,
            report_count=item.report_count,
            timestamp=item.timestamp,
        )

    def _build_observation(self) -> Observation:
        item = self.current_item
        assert item is not None
        return Observation(
            goal=self.task_config.goal,
            current_item=self._strip_item(item),
            queue_position=self.current_index + 1,
            queue_total=len(self.queue),
            reviewed_so_far=list(self.reviewed),
            context={
                "task": self.task_name,
                "difficulty": self.task_config.difficulty,
                "session_id": self.session_id,
                "steps_remaining": self.task_config.max_steps - self.step_count,
            },
            last_action_error=self.last_action_error,
        )

    def _build_reward(self, value: float, breakdown: Dict[str, float]) -> Reward:
        return Reward(
            value=value,
            breakdown=breakdown,
            cumulative=self.cumulative_reward,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_initial_observation(self) -> StepResult:
        return StepResult(
            observation=self._build_observation(),
            reward=Reward(value=0.0, breakdown={}, cumulative=0.0),
            done=False,
            info={
                "task": self.task_name,
                "queue_size": len(self.queue),
                "max_steps": self.task_config.max_steps,
                "session_id": self.session_id,
            },
        )

    def step(self, action: Action) -> StepResult:
        """Process one agent action and return the result."""

        # --- Validate episode state ---
        if self.done:
            return StepResult(
                observation=None,
                reward=self._build_reward(0.0, {}),
                done=True,
                info={"error": "Episode already done. Call /reset to start a new episode."},
            )

        # --- Validate item ID ---
        item = self.current_item
        if item is None:
            self.done = True
            return self._finalize()

        if action.target_item_id != item.item_id:
            self.last_action_error = (
                f"target_item_id '{action.target_item_id}' does not match "
                f"current item '{item.item_id}'. Action ignored."
            )
            return StepResult(
                observation=self._build_observation(),
                reward=self._build_reward(0.0, {"error": 0.0}),
                done=False,
                info={"error": self.last_action_error},
            )

        self.last_action_error = None
        self.step_count += 1

        # --- Compute reward ---
        step_reward, breakdown = compute_step_reward(action, item)
        self.cumulative_reward += step_reward

        # --- Record review ---
        reviewed = ReviewedItem(
            item_id=item.item_id,
            action_taken=action.action_type,
            policy_cited=action.policy_violated,
            reasoning=action.reasoning,
            confidence=action.confidence,
            reward_received=step_reward,
        )
        self.reviewed.append(reviewed)
        self.reviewed_items_with_actions.append((item, action.action_type))

        # --- Advance queue ---
        self.current_index += 1

        # --- Check termination ---
        max_steps_hit = self.step_count >= self.task_config.max_steps
        queue_exhausted = self.current_index >= len(self.queue)

        if queue_exhausted or max_steps_hit:
            return self._finalize(last_step_reward=step_reward, last_breakdown=breakdown)

        # --- Next observation ---
        reward_obj = self._build_reward(step_reward, breakdown)
        return StepResult(
            observation=self._build_observation(),
            reward=reward_obj,
            done=False,
            info={
                "step": self.step_count,
                "items_remaining": len(self.queue) - self.current_index,
            },
        )

    def _finalize(
        self,
        last_step_reward: float = 0.0,
        last_breakdown: Optional[Dict[str, float]] = None,
    ) -> StepResult:
        """Compute episode bonus, grade, and finalize."""
        self.done = True

        ep_bonus, ep_breakdown = compute_episode_bonus(
            reviewed_items=self.reviewed_items_with_actions,
            total_items=len(self.queue),
            steps_used=self.step_count,
            max_steps=self.task_config.max_steps,
        )
        self.cumulative_reward += ep_bonus

        # Grade
        grade_score, grade_breakdown = grade_episode(
            self.task_name,
            self.queue,
            self.reviewed,
        )

        normalized = normalize_score(
            self.cumulative_reward,
            self._min_possible,
            self._max_possible,
        )

        final_breakdown = dict(last_breakdown or {})
        final_breakdown.update({f"ep_{k}": v for k, v in ep_breakdown.items()})

        info: Dict[str, Any] = {
            "done": True,
            "steps_used": self.step_count,
            "items_reviewed": len(self.reviewed),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "normalized_score": round(normalized, 4),
            "grader_score": grade_score,
            "grader_breakdown": grade_breakdown,
            "episode_bonus": round(ep_bonus, 4),
            "session_id": self.session_id,
            "task": self.task_name,
        }

        return StepResult(
            observation=None,
            reward=Reward(
                value=last_step_reward + ep_bonus,
                breakdown=final_breakdown,
                cumulative=self.cumulative_reward,
            ),
            done=True,
            info=info,
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_name": self.task_name,
            "difficulty": self.task_config.difficulty,
            "step": self.step_count,
            "done": self.done,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "queue_size": len(self.queue),
            "current_index": self.current_index,
            "items_reviewed": len(self.reviewed),
            "reviewed": [r.model_dump() for r in self.reviewed],
            "current_item": (
                self._strip_item(self.current_item).model_dump()
                if self.current_item else None
            ),
        }


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

_sessions: Dict[str, Episode] = {}


def reset_episode(task_name: str, session_id: str = "default") -> StepResult:
    ep = Episode(task_name=task_name, session_id=session_id)
    _sessions[session_id] = ep
    return ep.get_initial_observation()


def step_episode(action: Action, session_id: str = "default") -> StepResult:
    ep = _sessions.get(session_id)
    if ep is None:
        return StepResult(
            observation=None,
            reward=Reward(value=0.0, breakdown={}, cumulative=0.0),
            done=True,
            info={"error": f"No active episode for session '{session_id}'. Call /reset first."},
        )
    return ep.step(action)


def get_state(session_id: str = "default") -> Dict[str, Any]:
    ep = _sessions.get(session_id)
    if ep is None:
        return {"error": f"No active episode for session '{session_id}'."}
    return ep.get_state()


def list_sessions() -> List[str]:
    return list(_sessions.keys())
