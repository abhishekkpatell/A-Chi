from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supportdesk_env.models import SupportAction
from supportdesk_env.server.environment import SupportDeskEnvironment
from supportdesk_env.task_bank import TASK_ORDER, TASKS


def main() -> None:
    env = SupportDeskEnvironment()
    assert len(TASK_ORDER) >= 3, "Need at least three tasks"

    for task_id in TASK_ORDER:
        reset_result = env.reset(task_id=task_id)
        assert reset_result.observation.task_id == task_id
        action = SupportAction(
            action_type="finalize",
            issue_type=TASKS[task_id].target_issue_type,
            priority=TASKS[task_id].target_priority,
            team=TASKS[task_id].target_team,
            severity=TASKS[task_id].target_severity,
            status=TASKS[task_id].target_status,
            tags=TASKS[task_id].required_tags,
            message=" ".join(TASKS[task_id].required_phrases),
            internal_note="validation run",
            confidence=0.9,
        )
        step_result = env.step(action)
        assert 0.0 <= step_result.reward.total <= 1.0
        assert 0.0 <= step_result.observation.current_score <= 1.0

    print(json.dumps({"status": "ok", "tasks": TASK_ORDER}, indent=2))


if __name__ == "__main__":
    main()
