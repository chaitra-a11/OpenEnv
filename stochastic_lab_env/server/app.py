"""FastAPI app for the stochastic lab environment."""

from pathlib import Path
import sys

from openenv.core.env_server.http_server import create_app

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

try:
    from ..models import StochasticLabAction, StochasticLabObservation
    from .stochastic_lab_environment import StochasticLabEnvironment
except ImportError:
    from stochastic_lab_env.models import StochasticLabAction, StochasticLabObservation
    from stochastic_lab_env.server.stochastic_lab_environment import (
        StochasticLabEnvironment,
    )


app = create_app(
    StochasticLabEnvironment,
    StochasticLabAction,
    StochasticLabObservation,
    env_name="stochastic_lab_env",
    max_concurrent_envs=4,
)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the local development server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """CLI entry point used by the generated `server` script."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
