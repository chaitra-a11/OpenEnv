"""Reference LLM inference script for the stochastic lab OpenEnv project."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

from openai import OpenAI

PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from stochastic_lab_env import StochasticLabAction, StochasticLabEnv
from stochastic_lab_env.baseline import choose_action as fallback_choose_action
from stochastic_lab_env.models import StochasticLabObservation, StochasticLabState


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")

TASK_NAME = os.getenv("STOCHASTIC_LAB_TASK", "adaptive-lab-scheduling")
BENCHMARK = os.getenv("STOCHASTIC_LAB_BENCHMARK", "stochastic_lab_env")
ENV_SEED = int(os.getenv("ENV_SEED", "7"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "14"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.55"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a stochastic laboratory scheduling environment.

    Objective:
    - maximize the final overall_score in [0, 1]
    - finish tasks before deadlines
    - avoid contamination and equipment failures

    Available actions:
    - process:<task_id>
    - delay:<task_id>
    - clean
    - idle

    Rules:
    - Return exactly one action on one line.
    - Do not add explanations, JSON, bullets, or markdown.
    - Use clean when contamination is risky.
    - Use idle when equipment is failed and needs repair.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def format_action(action: StochasticLabAction) -> str:
    if action.action_type in {"clean", "idle"}:
        return f"{action.action_type}()"
    return f"{action.action_type}('{action.task_id}')"


def build_task_summary(observation: StochasticLabObservation) -> str:
    lines: list[str] = []
    for task in observation.tasks:
        lines.append(
            (
                f"- {task.task_id}: status={task.status}, grade={task.grade}, "
                f"progress={task.completed_units:.2f}/{task.required_units:.2f}, "
                f"deadline={task.deadline}, delay_count={task.delay_count}, "
                f"completion_score={task.completion_score:.3f}"
            )
        )
    return "\n".join(lines)


def build_user_prompt(
    observation: StochasticLabObservation,
    state: StochasticLabState,
    step: int,
    history: list[str],
) -> str:
    history_text = "\n".join(history[-4:]) if history else "None"
    task_summary = build_task_summary(observation)
    return textwrap.dedent(
        f"""
        Step: {step}
        Current time: {observation.current_time}/{observation.time_horizon}
        Contamination probability: {observation.contamination_probability:.3f}
        Contamination active: {observation.contamination_active}
        Equipment status: {observation.equipment_status}
        Equipment health: {observation.equipment_health:.3f}
        Repair steps remaining: {observation.repair_steps_remaining}
        Current task_completion_score: {observation.task_completion_score:.3f}
        Current safety_score: {observation.safety_score:.3f}
        Current efficiency_score: {observation.efficiency_score:.3f}
        Current overall_score: {observation.overall_score:.3f}
        Recommended task: {observation.recommended_task_id}
        Last outcome: {observation.last_outcome or "None"}
        Last action error: {observation.last_action_error or "null"}

        Tasks:
        {task_summary}

        Recent history:
        {history_text}

        Choose the next action for this lab.
        Return exactly one of:
        process:<task_id>
        delay:<task_id>
        clean
        idle
        """
    ).strip()


def parse_action_text(
    raw_text: str,
    observation: StochasticLabObservation,
) -> StochasticLabAction | None:
    text = raw_text.strip().splitlines()[0].strip().lower()
    pending_task_ids = {
        task.task_id.lower(): task.task_id
        for task in observation.tasks
        if task.status == "pending"
    }

    if text == "clean":
        return StochasticLabAction(action_type="clean")
    if text == "idle":
        return StochasticLabAction(action_type="idle")

    match = re.fullmatch(r"(process|delay)\s*[:(]\s*([a-z0-9\-]+)\s*\)?", text)
    if not match:
        return None

    action_type, task_key = match.groups()
    task_id = pending_task_ids.get(task_key)
    if task_id is None:
        return None
    return StochasticLabAction(action_type=action_type, task_id=task_id)


def get_model_action(
    client: OpenAI,
    observation: StochasticLabObservation,
    state: StochasticLabState,
    step: int,
    history: list[str],
) -> StochasticLabAction:
    prompt = build_user_prompt(observation, state, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        parsed_action = parse_action_text(raw_text, observation)
        if parsed_action is not None:
            return parsed_action
    except Exception:
        pass

    return fallback_choose_action(state)


async def create_env() -> StochasticLabEnv:
    if LOCAL_IMAGE_NAME:
        return await StochasticLabEnv.from_docker_image(LOCAL_IMAGE_NAME)

    env = StochasticLabEnv(base_url=ENV_BASE_URL)
    await env.connect()
    return env


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing")

    env: StochasticLabEnv | None = None
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await create_env()
        result = await env.reset(seed=ENV_SEED)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            observation = result.observation
            state = await env.state()
            action = get_model_action(client, observation, state, step, history)

            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=format_action(action),
                reward=reward,
                done=result.done,
                error=observation.last_action_error,
            )

            history.append(
                f"step={step} action={format_action(action)} "
                f"reward={reward:.2f} score={observation.overall_score:.3f}"
            )

            if result.done:
                break

        final_observation = result.observation
        score = min(max(final_observation.overall_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        success = False

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
