"""Sensor-only contracts for the new task executor and pluggable controllers."""

from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np

from .types import Observation, ActionCommand
from .perception.pipeline import Detection


@dataclass
class ObjectTrack:
    track_id: str
    color: str
    position: np.ndarray
    yaw: float
    confidence: float
    status: str = "pending"
    attempts: int = 0
    visible: bool = True


@dataclass
class SceneEstimate:
    detections: list[Detection]
    time: float


@dataclass(frozen=True)
class TaskGoal:
    goal_id: str
    track_id: str
    color: str
    pick_position: np.ndarray
    pick_yaw: float
    slot_id: str
    place_position: np.ndarray
    instruction: str
    grasp_roll: float | None = None
    standoff: np.ndarray | None = None
    approach_height: float = 0.075
    lift_height: float = 0.10


@dataclass
class RuntimeContext:
    """No env, simulator object IDs, ground truth, contacts, or evaluation callback."""

    kinematics: object
    rgbd: Callable[[], dict]
    rest: np.ndarray
    limits: np.ndarray
    control_hz: float
    control_dt: float
    camera_fovy: float
    calibration: object
    seed: int = 0


@dataclass
class PolicyOutput:
    nominal: ActionCommand
    phase: str | None
    status: str = "running"


class BasePolicy(Protocol):
    supported_tasks: frozenset[str]

    def reset_episode(self, context: RuntimeContext) -> None: ...
    def start_goal(self, goal: TaskGoal, observation: Observation) -> None: ...
    def act(self, observation: Observation) -> PolicyOutput | None: ...


@dataclass
class ActionCorrection:
    delta: np.ndarray = field(default_factory=lambda: np.zeros(6, np.float32))
    source: str = "none"


class LocalOptimizer(Protocol):
    enabled_phases: frozenset[str]

    def reset(self) -> None: ...
    def optimize(
        self, observation: Observation, nominal: np.ndarray, goal: TaskGoal | None, phase: str | None
    ) -> ActionCorrection: ...


class NoOpOptimizer:
    enabled_phases = frozenset()

    def reset(self):
        pass

    def optimize(self, observation, nominal, goal, phase):
        return ActionCorrection()
