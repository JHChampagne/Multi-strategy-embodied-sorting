"""Base-policy selection is exclusive; local optimizers are registered separately."""

import json
from pathlib import Path
from types import SimpleNamespace

from ..contracts import PolicyOutput
from ..types import TaskSpec
from .base_geometric import GeometricBasePolicy


class LearnedBasePolicy:
    requires_grasp_plan = False

    def __init__(self, kind, checkpoint=None, device="cuda", task="sorting_basic"):
        self.task = task
        manifest = Path(checkpoint) / "task_capabilities.json" if checkpoint else None
        capabilities = json.loads(manifest.read_text()) if manifest and manifest.is_file() else {}
        self.supported_tasks = frozenset(capabilities.get("supported_tasks", ["pickplace_compat"]))
        if task not in self.supported_tasks:
            raise ValueError(
                f"{kind}: seven-material policy is not ready. Supply a task-trained checkpoint "
                "with task_capabilities.json; public M1 checkpoints only support pickplace_compat."
            )
        from .learned import LeRobotPolicy

        self.adapter = LeRobotPolicy("vla" if kind == "smolvla" else "act", checkpoint, device)

    def reset_episode(self, context):
        self.context = context
        self.goal = None
        self.adapter.reset(SimpleNamespace(seed=context.seed, task=TaskSpec(name=self.task)))

    def start_goal(self, goal, observation):
        self.goal = goal
        task = TaskSpec(
            name=self.task,
            material=goal.color,
            tray=goal.slot_id,
            target=goal.place_position,
            instruction=goal.instruction,
        )
        self.adapter.reset(SimpleNamespace(seed=self.context.seed, task=task))

    def act(self, observation):
        return PolicyOutput(self.adapter.act(observation), phase=None)


def make_base_policy(name="geometric", checkpoint=None, device="cuda", task="sorting_basic"):
    if name == "geometric":
        return GeometricBasePolicy()
    if name in {"act", "smolvla"}:
        return LearnedBasePolicy(name, checkpoint, device, task)
    raise ValueError(f"Unknown base policy: {name}")
