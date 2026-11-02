"""Public policy contracts. Policy observations never contain object ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, TypedDict

import numpy as np

JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
COLORS = ("red", "green", "yellow")
JAW_OPEN = 1.4
JAW_CLOSED = -0.05
BLOCK_HALF = 0.012
TRAY_HALF = 0.037
TRAYS = {"red": (-0.18, -0.13), "green": (0.18, -0.13), "yellow": (0.23, 0.0)}


class Observation(TypedDict):
    state: np.ndarray
    velocity: np.ndarray
    control: np.ndarray
    tcp: np.ndarray
    images: dict[str, np.ndarray]
    time: float
    task: str


@dataclass(frozen=True)
class TaskSpec:
    name: str = "sorting"
    material: str | None = None
    tray: str | None = None
    target: np.ndarray | None = None
    instruction: str = "Sort the red, green and yellow blocks into their assigned trays."


@dataclass
class ActionCommand:
    values: np.ndarray
    kind: Literal["absolute", "delta"] = "absolute"
    controller: str = "policy"
    phase: str = "execute"
    metadata: dict = field(default_factory=dict)

    def absolute(self, current: np.ndarray, limits: np.ndarray, delta_max: float = 0.05) -> np.ndarray:
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("Action must contain six finite joint commands")
        if self.kind == "delta":
            values = np.asarray(current) + np.clip(values, -1, 1) * delta_max
        elif self.kind != "absolute":
            raise ValueError(f"Unknown action kind: {self.kind}")
        return np.clip(values, limits[:, 0], limits[:, 1])


class Oracle(Protocol):
    def object_pose(self, color: str) -> tuple[np.ndarray, float]: ...


@dataclass
class PolicyContext:
    task: TaskSpec
    kinematics: object
    rgbd: Callable[[], dict]
    camera_fovy: float
    control_hz: float
    limits: np.ndarray
    rest: np.ndarray
    trays: dict[str, tuple[float, float]]
    calibration: object | None = None
    oracle: Oracle | None = None
    seed: int = 0
    contact_guard: Callable[[], list[str]] | None = None


class Policy(Protocol):
    supported_tasks: frozenset[str]

    def reset(self, context: PolicyContext) -> None: ...

    def act(self, observation: Observation) -> ActionCommand | None: ...
