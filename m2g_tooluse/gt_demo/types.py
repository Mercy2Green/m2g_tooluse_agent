from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Pose3D:
    """Pose in a named frame.

    Orientation is a unit quaternion in IsaacLab's ``wxyz`` order:
    ``(qw, qx, qy, qz)``.
    """

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    frame: str = "world"


@dataclass(frozen=True)
class ObjectInfo:
    name: str
    prim_path: str
    pose: Pose3D
    size: Optional[tuple[float, float, float]] = None
    confidence: float = 1.0
    source: str = "isaac_ground_truth"


@dataclass(frozen=True)
class GraspPlan:
    object_name: str
    pregrasp_pose: Pose3D
    grasp_pose: Pose3D
    lift_pose: Pose3D
    gripper_open: float
    gripper_closed: float
    strategy: str


@dataclass
class SkillResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
