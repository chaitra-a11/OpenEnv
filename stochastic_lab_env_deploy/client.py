"""Async client for the stochastic lab OpenEnv environment."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import StochasticLabAction, StochasticLabObservation, StochasticLabState


class StochasticLabEnv(
    EnvClient[StochasticLabAction, StochasticLabObservation, StochasticLabState]
):
    """Client for the stochastic lab scheduling environment."""

    def _step_payload(self, action: StochasticLabAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[StochasticLabObservation]:
        observation = StochasticLabObservation.model_validate(
            {
                **payload.get("observation", {}),
                "reward": payload.get("reward"),
                "done": payload.get("done", False),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> StochasticLabState:
        return StochasticLabState.model_validate(payload)

