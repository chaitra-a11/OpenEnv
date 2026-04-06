"""Deterministic heuristic baseline for the stochastic lab environment."""

from __future__ import annotations

import math
import json
from pathlib import Path
from statistics import mean

from .models import StochasticLabAction, StochasticLabState, TaskSnapshot
from .server.stochastic_lab_environment import StochasticLabEnvironment

DEFAULT_SEEDS = [3, 7, 11, 19, 23]


def choose_action(state: StochasticLabState) -> StochasticLabAction:
    pending = [task for task in state.tasks if task.status == "pending"]
    if not pending:
        return StochasticLabAction(action_type="idle")

    if state.contamination_active or state.contamination_probability > 0.68:
        return StochasticLabAction(action_type="clean")

    if state.equipment_status == "failed":
        return StochasticLabAction(action_type="idle")

    target = max(pending, key=lambda task: urgency_score(state, task))
    if state.equipment_health < 0.32 and target.grade == "hard":
        return StochasticLabAction(action_type="delay", task_id=target.task_id)

    if state.contamination_probability > 0.50 and target.grade != "easy":
        return StochasticLabAction(action_type="clean")

    return StochasticLabAction(action_type="process", task_id=target.task_id)


def urgency_score(state: StochasticLabState, task: TaskSnapshot) -> float:
    remaining = max(1, math.ceil(task.required_units - task.completed_units))
    slack = task.deadline - state.current_time - remaining
    grade_weight = {"easy": 1.0, "medium": 1.25, "hard": 1.5}[task.grade]
    return (0.10 * grade_weight) - float(slack)


def run_episode(seed: int) -> dict:
    env = StochasticLabEnvironment()
    observation = env.reset(seed=seed)
    history: list[dict] = []

    while not observation.done:
        action = choose_action(env.state)
        observation = env.step(action)
        history.append(
            {
                "time": env.state.current_time,
                "action": action.model_dump(exclude_none=True),
                "reward": observation.reward,
                "overall_score": observation.overall_score,
                "outcome": observation.last_outcome,
            }
        )

    return {
        "seed": seed,
        "cumulative_reward": env.state.cumulative_reward,
        "overall_score": env.state.overall_score,
        "task_completion_score": env.state.task_completion_score,
        "safety_score": env.state.safety_score,
        "efficiency_score": env.state.efficiency_score,
        "history": history,
        "tasks": [task.model_dump() for task in env.state.tasks],
    }


def run_baseline(seeds: list[int], save_path: Path | None = None) -> dict:
    episodes = [run_episode(seed) for seed in seeds]
    summary = {
        "seeds": seeds,
        "average_cumulative_reward": round(
            mean(episode["cumulative_reward"] for episode in episodes), 4
        ),
        "average_overall_score": round(
            mean(episode["overall_score"] for episode in episodes), 4
        ),
        "average_task_completion_score": round(
            mean(episode["task_completion_score"] for episode in episodes), 4
        ),
        "average_safety_score": round(
            mean(episode["safety_score"] for episode in episodes), 4
        ),
        "average_efficiency_score": round(
            mean(episode["efficiency_score"] for episode in episodes), 4
        ),
        "episodes": episodes,
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
