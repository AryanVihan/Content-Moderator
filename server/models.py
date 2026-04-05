from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class UserHistory(BaseModel):
    account_age_days: int
    prior_violations: int
    follower_count: int
    is_verified: bool
    is_satire_account: bool


class ContentItem(BaseModel):
    item_id: str
    platform: Literal["facebook", "instagram", "whatsapp", "marketplace"]
    content_text: str
    user_history: UserHistory
    report_count: int
    timestamp: str
    # ground_truth fields are present but excluded from agent observation
    ground_truth_action: str = Field(exclude=True)
    ground_truth_policy: Optional[str] = Field(default=None, exclude=True)
    difficulty_notes: str = Field(default="", exclude=True)
    # Extra flags used by grader
    is_urgent: bool = Field(default=False, exclude=True)
    is_evasion: bool = Field(default=False, exclude=True)
    is_coordinated: bool = Field(default=False, exclude=True)
    needs_context: bool = Field(default=False, exclude=True)


class AgentVisibleContentItem(BaseModel):
    """Content item stripped of ground truth — what the agent actually sees."""
    item_id: str
    platform: str
    content_text: str
    user_history: UserHistory
    report_count: int
    timestamp: str


class ReviewedItem(BaseModel):
    item_id: str
    action_taken: str
    policy_cited: Optional[str]
    reasoning: str
    confidence: float
    reward_received: float


class Observation(BaseModel):
    goal: str
    current_item: AgentVisibleContentItem
    queue_position: int
    queue_total: int
    reviewed_so_far: List[ReviewedItem]
    context: Dict[str, Any]
    last_action_error: Optional[str] = None


class Action(BaseModel):
    action_type: Literal[
        "REMOVE",
        "KEEP",
        "ESCALATE",
        "ADD_WARNING_LABEL",
        "REQUEST_CONTEXT",
    ]
    target_item_id: str
    policy_violated: Optional[str] = None
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]
    cumulative: float


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: Dict[str, Any]


class EnvironmentState(BaseModel):
    session_id: str
    task_name: str
    step: int
    done: bool
    cumulative_reward: float
    queue: List[Dict[str, Any]]
    reviewed: List[ReviewedItem]
    current_index: int
