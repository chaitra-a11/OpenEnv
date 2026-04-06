"""Typed models for the stochastic lab scheduling environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator

TaskGrade = Literal["easy", "medium", "hard"]
EquipmentStatus = Literal["operational", "degraded", "failed"]
ActionKind = Literal["process", "delay", "clean", "idle"]


class RewardBreakdown(BaseModel):
    """Explain how the step reward was assembled."""

    progress: float = Field(default=0.0)
    completion: float = Field(default=0.0)
    safety: float = Field(default=0.0)
    efficiency: float = Field(default=0.0)
    lateness: float = Field(default=0.0)
    event_penalty: float = Field(default=0.0)
    total: float = Field(default=0.0)


class TaskSnapshot(BaseModel):
    """Serializable task view shared by state and observation."""

    task_id: str = Field(..., description="Stable task identifier.")
    title: str = Field(..., description="Human-readable task name.")
    grade: TaskGrade = Field(..., description="Difficulty band for the task.")
    required_units: float = Field(..., gt=0.0, description="Total work required.")
    completed_units: float = Field(
        default=0.0, ge=0.0, description="Work already completed."
    )
    progress_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Normalized progress between 0 and 1."
    )
    deadline: int = Field(..., ge=0, description="Absolute time step deadline.")
    contamination_sensitivity: float = Field(
        ..., ge=0.0, le=1.0, description="How strongly contamination hurts quality."
    )
    equipment_stress: float = Field(
        ..., ge=0.0, le=1.0, description="How much processing stresses equipment."
    )
    delay_count: int = Field(default=0, ge=0, description="Number of explicit delays.")
    status: Literal["pending", "completed"] = Field(default="pending")
    completion_time: int | None = Field(default=None, ge=0)
    completion_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final normalized score for this task once complete.",
    )
    deadline_missed: bool = Field(default=False)


class StochasticLabAction(Action):
    """Action for the lab scheduler."""

    action_type: ActionKind = Field(..., description="Scheduling action to execute.")
    task_id: str | None = Field(
        default=None,
        description="Task identifier for process/delay actions. Not used for clean/idle.",
    )

    @model_validator(mode="after")
    def validate_task_requirements(self) -> "StochasticLabAction":
        if self.action_type in {"process", "delay"} and not self.task_id:
            raise ValueError("task_id is required for process and delay actions.")
        if self.action_type in {"clean", "idle"} and self.task_id is not None:
            raise ValueError("task_id must be omitted for clean and idle actions.")
        return self


class StochasticLabObservation(Observation):
    """Observation returned after each environment transition."""

    current_time: int = Field(default=0, ge=0)
    time_horizon: int = Field(default=0, ge=1)
    contamination_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    contamination_active: bool = Field(default=False)
    equipment_status: EquipmentStatus = Field(default="operational")
    equipment_health: float = Field(default=1.0, ge=0.0, le=1.0)
    repair_steps_remaining: int = Field(default=0, ge=0)
    tasks: list[TaskSnapshot] = Field(default_factory=list)
    pending_task_count: int = Field(default=0, ge=0)
    completed_task_count: int = Field(default=0, ge=0)
    task_completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_score: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_action: str = Field(default="")
    last_outcome: str = Field(default="")
    last_action_error: str | None = Field(default=None)
    recommended_task_id: str | None = Field(default=None)
    recent_events: list[str] = Field(default_factory=list)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)


class StochasticLabState(State):
    """Internal mutable state for stochastic lab operations."""

    current_time: int = Field(default=0, ge=0)
    time_horizon: int = Field(default=12, ge=1)
    contamination_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    contamination_active: bool = Field(default=False)
    equipment_status: EquipmentStatus = Field(default="operational")
    equipment_health: float = Field(default=0.95, ge=0.0, le=1.0)
    repair_steps_remaining: int = Field(default=0, ge=0)
    tasks: list[TaskSnapshot] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    task_completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_score: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int | None = Field(default=None)
    incident_count: int = Field(default=0, ge=0)
    missed_deadlines: int = Field(default=0, ge=0)
    clean_count: int = Field(default=0, ge=0)
    idle_count: int = Field(default=0, ge=0)
    delay_count: int = Field(default=0, ge=0)
    process_count: int = Field(default=0, ge=0)
    average_contamination: float = Field(default=0.0, ge=0.0, le=1.0)
    last_action: str = Field(default="")
    last_outcome: str = Field(default="")
    last_action_error: str | None = Field(default=None)
    recent_events: list[str] = Field(default_factory=list)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
