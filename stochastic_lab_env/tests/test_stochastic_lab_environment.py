from stochastic_lab_env.baseline import run_baseline
from stochastic_lab_env.models import StochasticLabAction
from stochastic_lab_env.server.stochastic_lab_environment import StochasticLabEnvironment


def test_reset_creates_three_graded_tasks() -> None:
    env = StochasticLabEnvironment()
    observation = env.reset(seed=7)

    grades = [task.grade for task in observation.tasks]
    assert grades == ["easy", "medium", "hard"]
    assert observation.pending_task_count == 3
    assert observation.overall_score >= 0.0


def test_process_action_advances_progress() -> None:
    env = StochasticLabEnvironment()
    env.reset(seed=11)
    easy_task = next(task for task in env.state.tasks if task.grade == "easy")

    result = env.step(
        StochasticLabAction(action_type="process", task_id=easy_task.task_id)
    )

    updated_task = next(task for task in env.state.tasks if task.task_id == easy_task.task_id)
    assert updated_task.completed_units > 0.0
    assert result.reward is not None


def test_clean_reduces_contamination_probability() -> None:
    env = StochasticLabEnvironment()
    env.reset(seed=13)
    env.state.contamination_probability = 0.82
    env.state.contamination_active = True

    env.step(StochasticLabAction(action_type="clean"))

    assert env.state.contamination_probability < 0.82


def test_baseline_is_reproducible_for_fixed_seeds() -> None:
    first = run_baseline([3, 7], save_path=None)
    second = run_baseline([3, 7], save_path=None)

    assert first["average_overall_score"] == second["average_overall_score"]
    assert [
        episode["overall_score"] for episode in first["episodes"]
    ] == [episode["overall_score"] for episode in second["episodes"]]
