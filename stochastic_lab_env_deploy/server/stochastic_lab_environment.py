"""Stochastic OpenEnv environment for adaptive lab scheduling."""

from __future__ import annotations

import math
import random
import sys
from typing import Any
from uuid import uuid4
from pathlib import Path

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

try:
    from ..models import (
        ActionKind,
        RewardBreakdown,
        StochasticLabAction,
        StochasticLabObservation,
        StochasticLabState,
        TaskGrade,
        TaskSnapshot,
    )
except ImportError:
    from stochastic_lab_env.models import (
        ActionKind,
        RewardBreakdown,
        StochasticLabAction,
        StochasticLabObservation,
        StochasticLabState,
        TaskGrade,
        TaskSnapshot,
    )


GRADE_WEIGHT: dict[TaskGrade, float] = {
    "easy": 1.0,
    "medium": 1.25,
    "hard": 1.5,
}

TASK_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "task_id": "easy-buffer-prep",
        "title": "Buffer Preparation",
        "grade": "easy",
        "required_units": 2.0,
        "deadline_base": 4,
        "contamination_sensitivity": 0.20,
        "equipment_stress": 0.18,
    },
    {
        "task_id": "medium-culture-refresh",
        "title": "Cell Culture Refresh",
        "grade": "medium",
        "required_units": 3.0,
        "deadline_base": 6,
        "contamination_sensitivity": 0.45,
        "equipment_stress": 0.30,
    },
    {
        "task_id": "hard-sterile-transfer",
        "title": "Sterile Assay Transfer",
        "grade": "hard",
        "required_units": 4.5,
        "deadline_base": 8,
        "contamination_sensitivity": 0.75,
        "equipment_stress": 0.48,
    },
)


class StochasticLabEnvironment(
    Environment[StochasticLabAction, StochasticLabObservation, StochasticLabState]
):
    """Scheduling environment with deadlines, contamination, and equipment failures."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random()
        self._state = StochasticLabState(episode_id=str(uuid4()))

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        time_horizon: int | None = None,
        **kwargs: Any,
    ) -> StochasticLabObservation:
        del kwargs

        actual_seed = (
            seed if seed is not None else random.SystemRandom().randrange(1, 10**9)
        )
        self._rng = random.Random(actual_seed)

        tasks = [self._build_task(template) for template in TASK_TEMPLATES]
        computed_horizon = time_horizon or max(task.deadline for task in tasks) + 4
        initial_contamination = round(self._rng.uniform(0.08, 0.16), 3)

        self._state = StochasticLabState(
            episode_id=episode_id or str(uuid4()),
            current_time=0,
            time_horizon=computed_horizon,
            contamination_probability=initial_contamination,
            contamination_active=False,
            equipment_status="operational",
            equipment_health=0.95,
            repair_steps_remaining=0,
            tasks=tasks,
            cumulative_reward=0.0,
            seed=actual_seed,
            recent_events=["Episode initialized with three lab tasks."],
            last_action="reset",
            last_outcome="Lab ready. Prioritize tasks while managing contamination risk.",
            last_action_error=None,
        )
        self._refresh_scores()
        self._state.reward_breakdown = RewardBreakdown(total=0.0)
        return self._build_observation(0.0, done=False)

    def step(
        self,
        action: StochasticLabAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> StochasticLabObservation:
        del timeout_s, kwargs

        if self._is_terminal():
            raise RuntimeError("Episode is complete. Call reset() before step().")

        breakdown = RewardBreakdown()
        self._state.last_action = action.action_type
        self._state.last_outcome = ""
        self._state.last_action_error = None

        task = self._get_task(action.task_id) if action.task_id else None
        if action.action_type == "process":
            self._process_task(task, breakdown)
        elif action.action_type == "delay":
            self._delay_task(task, breakdown)
        elif action.action_type == "clean":
            self._clean_lab(breakdown)
        else:
            self._idle(breakdown)

        self._apply_stochastic_events(action.action_type, task, breakdown)
        self._advance_time(breakdown)
        self._clip_state()
        self._refresh_scores()

        done = self._is_terminal()
        if done:
            self._apply_terminal_adjustments(breakdown)
            self._refresh_scores()

        breakdown.total = round(
            breakdown.progress
            + breakdown.completion
            + breakdown.safety
            + breakdown.efficiency
            + breakdown.lateness
            + breakdown.event_penalty,
            4,
        )
        self._state.reward_breakdown = breakdown
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + breakdown.total, 4
        )

        return self._build_observation(breakdown.total, done=done)

    @property
    def state(self) -> StochasticLabState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="StochasticLabEnvironment",
            description=(
                "Adaptive scheduling environment for stochastic laboratory operations "
                "with deadlines, contamination events, and equipment failure."
            ),
            version="0.1.0",
        )

    def _build_task(self, template: dict[str, Any]) -> TaskSnapshot:
        deadline = template["deadline_base"] + self._rng.randint(0, 1)
        return TaskSnapshot(
            task_id=template["task_id"],
            title=template["title"],
            grade=template["grade"],
            required_units=template["required_units"],
            completed_units=0.0,
            progress_fraction=0.0,
            deadline=deadline,
            contamination_sensitivity=template["contamination_sensitivity"],
            equipment_stress=template["equipment_stress"],
        )

    def _build_observation(
        self, reward: float, *, done: bool
    ) -> StochasticLabObservation:
        pending_tasks = self._pending_tasks()
        return StochasticLabObservation(
            current_time=self._state.current_time,
            time_horizon=self._state.time_horizon,
            contamination_probability=round(self._state.contamination_probability, 3),
            contamination_active=self._state.contamination_active,
            equipment_status=self._state.equipment_status,
            equipment_health=round(self._state.equipment_health, 3),
            repair_steps_remaining=self._state.repair_steps_remaining,
            tasks=[task.model_copy(deep=True) for task in self._state.tasks],
            pending_task_count=len(pending_tasks),
            completed_task_count=len(self._completed_tasks()),
            task_completion_score=self._state.task_completion_score,
            safety_score=self._state.safety_score,
            efficiency_score=self._state.efficiency_score,
            overall_score=self._state.overall_score,
            last_action=self._state.last_action,
            last_outcome=self._state.last_outcome,
            last_action_error=self._state.last_action_error,
            recommended_task_id=self._recommend_task_id(),
            recent_events=list(self._state.recent_events),
            reward_breakdown=self._state.reward_breakdown.model_copy(deep=True),
            reward=reward,
            done=done,
        )

    def _process_task(
        self, task: TaskSnapshot | None, breakdown: RewardBreakdown
    ) -> None:
        if task is None:
            breakdown.efficiency -= 0.12
            self._state.last_action_error = "invalid_task_id"
            self._state.last_outcome = "Process action ignored because the task_id was invalid."
            return
        if task.status == "completed":
            breakdown.efficiency -= 0.10
            self._state.last_action_error = "task_already_completed"
            self._state.last_outcome = f"{task.title} is already completed."
            return
        if self._state.equipment_status == "failed":
            breakdown.event_penalty -= 0.40
            self._state.last_action_error = "equipment_failed"
            self._state.last_outcome = (
                "Processing failed because equipment is down. Use idle to recover."
            )
            return

        throughput = 1.0 if self._state.equipment_status == "operational" else 0.75
        if self._state.contamination_active:
            throughput *= 0.72

        remaining = max(0.0, task.required_units - task.completed_units)
        progress_units = min(remaining, throughput)
        task.completed_units = round(task.completed_units + progress_units, 3)
        task.progress_fraction = round(task.completed_units / task.required_units, 3)
        self._state.process_count += 1

        breakdown.progress += 0.40 * (progress_units / task.required_units)
        self._state.contamination_probability += (
            0.03 + 0.12 * task.contamination_sensitivity
        )
        self._state.equipment_health -= 0.03 + 0.07 * task.equipment_stress

        if task.progress_fraction >= 1.0 - 1e-6:
            task.status = "completed"
            task.completion_time = self._state.current_time + 1
            task.progress_fraction = 1.0
            task.completion_score = self._score_completed_task(task)
            breakdown.completion += 0.85 * task.completion_score
            self._state.last_outcome = (
                f"Completed {task.title} with task score {task.completion_score:.2f}."
            )
            self._log_event(
                f"{task.title} completed at time {self._state.current_time + 1}."
            )
        else:
            self._state.last_outcome = (
                f"Processed {task.title}: progress {task.completed_units:.2f}/"
                f"{task.required_units:.2f}."
            )

    def _delay_task(
        self, task: TaskSnapshot | None, breakdown: RewardBreakdown
    ) -> None:
        if task is None:
            breakdown.efficiency -= 0.08
            self._state.last_action_error = "invalid_task_id"
            self._state.last_outcome = "Delay action ignored because the task_id was invalid."
            return
        if task.status == "completed":
            breakdown.efficiency -= 0.05
            self._state.last_action_error = "task_already_completed"
            self._state.last_outcome = f"{task.title} is already completed."
            return

        task.delay_count += 1
        self._state.delay_count += 1
        remaining_units = max(0.0, task.required_units - task.completed_units)
        slack = task.deadline - self._state.current_time - math.ceil(remaining_units)
        if self._state.contamination_active or self._state.equipment_health < 0.35:
            breakdown.safety += 0.05 if task.grade != "easy" else 0.02
            breakdown.efficiency -= 0.02
            self._state.last_outcome = (
                f"Delayed {task.title} to reduce immediate safety risk."
            )
        else:
            breakdown.efficiency -= 0.03 + (0.03 if slack <= 1 else 0.01)
            self._state.last_outcome = f"Deferred {task.title}; slack remaining is {slack}."

        self._state.contamination_probability -= 0.03
        self._state.equipment_health += 0.02

    def _clean_lab(self, breakdown: RewardBreakdown) -> None:
        before = self._state.contamination_probability
        self._state.clean_count += 1
        self._state.contamination_probability = max(
            0.0, self._state.contamination_probability - 0.45
        )
        if self._state.contamination_active:
            success_threshold = 0.92 if self._state.equipment_status != "failed" else 0.75
            if self._rng.random() < success_threshold:
                self._state.contamination_active = False
                self._log_event("Cleaning cleared the active contamination event.")
            breakdown.safety += 0.16 + 0.16 * (before - self._state.contamination_probability)
            self._state.last_outcome = "Deep cleaning reduced contamination and improved safety."
        else:
            breakdown.safety += 0.04 if before > 0.30 else 0.0
            self._state.last_outcome = "Preventive cleaning reduced contamination risk."

        breakdown.efficiency -= 0.05
        self._state.equipment_health += 0.04

    def _idle(self, breakdown: RewardBreakdown) -> None:
        self._state.idle_count += 1
        if self._state.equipment_status == "failed":
            self._state.repair_steps_remaining = max(
                0, self._state.repair_steps_remaining - 1
            )
            if self._state.repair_steps_remaining == 0:
                self._state.equipment_status = "degraded"
                self._state.equipment_health = max(self._state.equipment_health, 0.45)
                self._log_event("Equipment recovered from failed to degraded after repair.")
                breakdown.safety += 0.10
            self._state.last_outcome = "Idle used for repair and stabilization."
        else:
            self._state.equipment_health += 0.03
            self._state.contamination_probability -= 0.02
            self._state.last_outcome = "Idle step conserved resources but cost schedule time."

        breakdown.efficiency -= 0.03

    def _apply_stochastic_events(
        self,
        action_type: ActionKind,
        task: TaskSnapshot | None,
        breakdown: RewardBreakdown,
    ) -> None:
        event_messages: list[str] = []
        if self._state.equipment_status != "failed":
            failure_probability = 0.005 + (1.0 - self._state.equipment_health) * 0.10
            if action_type == "process" and task is not None:
                failure_probability += 0.03 + 0.05 * task.equipment_stress
            if self._state.contamination_active:
                failure_probability += 0.03
            if self._rng.random() < min(0.85, failure_probability):
                self._state.equipment_status = "failed"
                self._state.repair_steps_remaining = 2 + self._rng.randint(0, 1)
                self._state.equipment_health = max(0.12, self._state.equipment_health - 0.20)
                self._state.incident_count += 1
                breakdown.event_penalty -= 0.35
                event_messages.append("Unexpected equipment failure disrupted operations.")
            elif (
                self._state.equipment_status == "operational"
                and self._state.equipment_health < 0.45
                and self._rng.random() < 0.35
            ):
                self._state.equipment_status = "degraded"
                breakdown.event_penalty -= 0.05
                event_messages.append("Equipment degraded and now processes work more slowly.")

        contamination_event_probability = self._state.contamination_probability * 0.45
        if action_type == "process":
            contamination_event_probability += 0.04
        if (
            not self._state.contamination_active
            and self._rng.random() < min(0.90, contamination_event_probability)
        ):
            self._state.contamination_active = True
            self._state.contamination_probability = min(
                1.0, self._state.contamination_probability + 0.12
            )
            self._state.incident_count += 1
            breakdown.event_penalty -= 0.25
            event_messages.append("Contamination event triggered additional safety overhead.")
        elif self._state.contamination_active and action_type != "clean":
            self._state.contamination_probability = min(
                1.0, self._state.contamination_probability + 0.02
            )

        for message in event_messages:
            self._log_event(message)
        if event_messages:
            self._state.last_outcome = " ".join(event_messages)

    def _advance_time(self, breakdown: RewardBreakdown) -> None:
        self._state.current_time += 1

        pending_overdue = 0
        for task in self._pending_tasks():
            overdue_steps = max(0, self._state.current_time - task.deadline)
            if overdue_steps > 0:
                weighted_penalty = 0.03 * overdue_steps * GRADE_WEIGHT[task.grade]
                breakdown.lateness -= weighted_penalty
                pending_overdue += 1
                if not task.deadline_missed:
                    task.deadline_missed = True
                    self._state.missed_deadlines += 1
                    self._log_event(f"{task.title} missed its deadline at time {self._state.current_time}.")

        if pending_overdue and not self._state.last_outcome:
            self._state.last_outcome = f"{pending_overdue} pending task(s) are now overdue."

        if not self._state.contamination_active:
            self._state.contamination_probability = max(
                0.0, self._state.contamination_probability - 0.01
            )

        contamination_history = (
            self._state.average_contamination * max(0, self._state.current_time - 1)
            + self._state.contamination_probability
        )
        self._state.average_contamination = contamination_history / self._state.current_time

    def _apply_terminal_adjustments(self, breakdown: RewardBreakdown) -> None:
        incomplete_count = len(self._pending_tasks())
        if incomplete_count:
            breakdown.lateness -= 0.10 * incomplete_count
        breakdown.completion += 0.30 * self._state.overall_score
        if self._state.overall_score >= 0.8:
            self._log_event("Episode ended with a strong scheduling policy.")
        elif incomplete_count:
            self._log_event("Episode ended with unfinished work under constrained resources.")

    def _score_completed_task(self, task: TaskSnapshot) -> float:
        completion_time = task.completion_time or self._state.current_time + 1
        overdue_steps = max(0, completion_time - task.deadline)
        timeliness = max(0.0, 1.0 - 0.12 * overdue_steps - 0.04 * task.delay_count)

        quality_penalty = self._state.contamination_probability * task.contamination_sensitivity
        if self._state.contamination_active:
            quality_penalty += 0.15 * task.contamination_sensitivity
        if self._state.equipment_status == "degraded":
            quality_penalty += 0.10 * task.equipment_stress
        if self._state.equipment_status == "failed":
            quality_penalty += 0.25
        quality = max(0.0, 1.0 - quality_penalty)

        score = max(0.0, min(1.0, 0.55 * timeliness + 0.45 * quality))
        return round(score, 3)

    def _refresh_scores(self) -> None:
        task_scores = [
            max(task.completion_score, 0.40 * task.progress_fraction)
            for task in self._state.tasks
        ]
        self._state.task_completion_score = round(
            sum(task_scores) / max(1, len(task_scores)), 3
        )

        safety = 1.0
        safety -= 0.08 * self._state.incident_count
        safety -= 0.18 * self._state.average_contamination
        if self._state.contamination_active:
            safety -= 0.05
        if self._state.equipment_status == "failed":
            safety -= 0.08
        self._state.safety_score = round(max(0.0, min(1.0, safety)), 3)

        efficiency = 1.0
        efficiency -= 0.35 * (
            self._state.current_time / max(1, self._state.time_horizon)
        )
        efficiency -= 0.02 * self._state.idle_count
        efficiency -= 0.015 * self._state.clean_count
        efficiency -= 0.015 * self._state.delay_count
        efficiency -= 0.03 * self._state.missed_deadlines
        self._state.efficiency_score = round(max(0.0, min(1.0, efficiency)), 3)

        overall = (
            0.65 * self._state.task_completion_score
            + 0.20 * self._state.safety_score
            + 0.15 * self._state.efficiency_score
        )
        self._state.overall_score = round(max(0.0, min(1.0, overall)), 3)

    def _recommend_task_id(self) -> str | None:
        pending_tasks = self._pending_tasks()
        if not pending_tasks:
            return None
        return max(
            pending_tasks,
            key=lambda task: self._urgency_score(task),
        ).task_id

    def _urgency_score(self, task: TaskSnapshot) -> float:
        remaining = max(1, math.ceil(task.required_units - task.completed_units))
        slack = task.deadline - self._state.current_time - remaining
        return (0.10 * GRADE_WEIGHT[task.grade]) - float(slack)

    def _pending_tasks(self) -> list[TaskSnapshot]:
        return [task for task in self._state.tasks if task.status == "pending"]

    def _completed_tasks(self) -> list[TaskSnapshot]:
        return [task for task in self._state.tasks if task.status == "completed"]

    def _get_task(self, task_id: str | None) -> TaskSnapshot | None:
        if task_id is None:
            return None
        for task in self._state.tasks:
            if task.task_id == task_id:
                return task
        return None

    def _clip_state(self) -> None:
        self._state.contamination_probability = max(
            0.0, min(1.0, self._state.contamination_probability)
        )
        self._state.equipment_health = max(0.0, min(1.0, self._state.equipment_health))
        if self._state.equipment_status == "operational" and self._state.equipment_health < 0.40:
            self._state.equipment_status = "degraded"
        if self._state.equipment_status == "degraded" and self._state.equipment_health > 0.75:
            self._state.equipment_status = "operational"

    def _log_event(self, message: str) -> None:
        recent = [*self._state.recent_events, message]
        self._state.recent_events = recent[-5:]

    def _is_terminal(self) -> bool:
        all_complete = all(task.status == "completed" for task in self._state.tasks)
        horizon_reached = self._state.current_time >= self._state.time_horizon
        return all_complete or horizon_reached
