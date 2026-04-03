from supportdesk_env.models import SupportAction
from supportdesk_env.server.environment import SupportDeskEnvironment
from supportdesk_env.task_bank import TASK_ORDER, TASKS


def test_three_tasks_exist() -> None:
    assert len(TASK_ORDER) >= 3


def test_reset_and_step_bounds() -> None:
    env = SupportDeskEnvironment()
    for task_id in TASK_ORDER:
        reset = env.reset(task_id=task_id)
        assert reset.observation.task_id == task_id
        action = SupportAction(
            action_type="finalize",
            issue_type=TASKS[task_id].target_issue_type,
            priority=TASKS[task_id].target_priority,
            team=TASKS[task_id].target_team,
            severity=TASKS[task_id].target_severity,
            status=TASKS[task_id].target_status,
            tags=TASKS[task_id].required_tags,
            message=" ".join(TASKS[task_id].required_phrases),
            internal_note="pytest",
            confidence=0.9,
        )
        result = env.step(action)
        assert 0.0 <= result.reward.total <= 1.0
        assert 0.0 <= result.observation.current_score <= 1.0
        assert result.done in {True, False}
