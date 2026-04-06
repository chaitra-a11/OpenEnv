---
title: Stochastic Lab Environment Server
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Stochastic Lab Environment

`stochastic_lab_env` is an OpenEnv reinforcement learning environment for stochastic
laboratory scheduling. An agent must balance task completion, contamination
control, equipment health, and deadline pressure under real-world uncertainty.

Live Space:

- [hanchinalchaitra/stochastic-lab-env](https://huggingface.co/spaces/hanchinalchaitra/stochastic-lab-env)
- Runtime host: [hanchinalchaitra-stochastic-lab-env.hf.space](https://hanchinalchaitra-stochastic-lab-env.hf.space)

## Environment Description

The environment models a small lab scheduler operating under uncertainty. Each
episode starts with three tasks spanning `easy`, `medium`, and `hard` grades,
with different workload sizes, contamination sensitivities, and equipment
stress profiles. The agent must learn adaptive policies that finish work on
time while avoiding contamination escalation and equipment downtime.

Real-world stochasticity is introduced through:

- random contamination events
- random equipment degradation and failure
- deadline pressure across tasks with different urgency profiles
- resource tradeoffs between throughput, cleaning, deferral, and recovery

## Action Space

The agent chooses one of four typed actions each step:

- `process(task_id)`: spend one step advancing a specific task
- `delay(task_id)`: intentionally defer a task to reduce immediate risk
- `clean`: reduce contamination probability and clear active contamination
- `idle`: conserve resources or recover failed equipment

## Observation Space

Each observation includes:

- `current_time` and `time_horizon`
- `contamination_probability` and `contamination_active`
- `equipment_status`, `equipment_health`, and `repair_steps_remaining`
- the full task list with progress, deadlines, grades, and completion scores
- aggregate `task_completion_score`, `safety_score`, `efficiency_score`, and `overall_score`
- recent events, last action outcome, recommended next task, and reward breakdown

## Reward Design

The reward is shaped with learnable intermediate signals:

- partial progress reward for useful work
- task completion reward scaled by normalized task score in `[0, 1]`
- safety reward for cleaning or delaying under risky conditions
- efficiency penalty for unnecessary idling or deferrals
- lateness penalty for overdue work
- event penalty for contamination spikes and equipment failures

Task scores are normalized in `[0, 1]` and incorporate timeliness, contamination
exposure, and equipment condition at completion. Each observation also reports:

- `task_completion_score`
- `safety_score`
- `efficiency_score`
- `overall_score`

## Project Layout

```text
stochastic_lab_env/
|-- Dockerfile
|-- __init__.py
|-- baseline.py
|-- client.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- README.md
|-- validate-submission.sh
|-- outputs/
|   |-- evals/
|   `-- logs/
|-- scripts/
|   `-- baseline_inference.py
|-- server/
|   |-- __init__.py
|   |-- app.py
|   |-- stochastic_lab_environment.py
|   |-- Dockerfile
|   `-- requirements.txt
`-- tests/
    |-- conftest.py
    `-- test_stochastic_lab_environment.py
```

## Local Setup

From the workspace root:

```bash
cd stochastic_lab_env
pip install -e .[dev]
```

Required environment variables for `inference.py`:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`
- `LOCAL_IMAGE_NAME` when using `from_docker_image(...)`

## Run Locally

```bash
cd stochastic_lab_env
server --port 8000
```

You can also run with Uvicorn directly:

```bash
cd stochastic_lab_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Baseline Inference

The heuristic baseline uses deterministic seeds and writes a reproducible summary
to `outputs/evals/baseline_scores.json`.

```bash
cd stochastic_lab_env
python scripts/baseline_inference.py
```

## LLM Inference Script

The project root also includes `inference.py`, which follows the required
`[START]`, `[STEP]`, and `[END]` stdout contract and uses the OpenAI client for
model calls.

Set these variables before running:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME` if using `from_docker_image(...)`

Optional variables:

- `ENV_BASE_URL` for a locally running server, default `http://127.0.0.1:8000`
- `ENV_SEED` for deterministic resets

Run it with:

```bash
cd stochastic_lab_env
python inference.py
```

## Example Client Usage

```python
import asyncio

from stochastic_lab_env import StochasticLabAction, StochasticLabEnv


async def main() -> None:
    async with StochasticLabEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(seed=7)
        while not result.done:
            state = await env.state()
            pending = [task for task in state.tasks if task.status == "pending"]
            if state.contamination_active:
                action = StochasticLabAction(action_type="clean")
            elif pending:
                action = StochasticLabAction(
                    action_type="process",
                    task_id=pending[0].task_id,
                )
            else:
                action = StochasticLabAction(action_type="idle")
            result = await env.step(action)
            print(result.observation.last_outcome)


asyncio.run(main())
```

## Validate

```bash
cd stochastic_lab_env
openenv validate
```

## Docker Build

The repository root includes a Hugging Face Spaces-ready [Dockerfile](/C:/Users/Chaitra%20Hanchinal/OneDrive/Desktop/hackOpenEnv/stochastic_lab_env/Dockerfile).

Build it locally with:

```bash
cd stochastic_lab_env
docker build .
```

## Submission Pre-Check

The project root includes [validate-submission.sh](/C:/Users/Chaitra%20Hanchinal/OneDrive/Desktop/hackOpenEnv/stochastic_lab_env/validate-submission.sh),
which checks:

- your Hugging Face Space responds to `/reset`
- the Docker image builds
- `openenv validate` passes

Example:

```bash
cd stochastic_lab_env
bash validate-submission.sh https://your-space.hf.space .
```

On Windows, run it from Git Bash or WSL.

## Deploy

```bash
cd stochastic_lab_env
openenv push
```

To deploy to a specific Hugging Face Space:

```bash
cd stochastic_lab_env
openenv push --repo-id <username>/<space-name>
```

If you want to validate the live Space after deployment:

```bash
cd stochastic_lab_env
bash validate-submission.sh https://<username>-<space-name>.hf.space .
```

When you share your sample inference script and pre-validation script, they can be
integrated without changing the core API.
