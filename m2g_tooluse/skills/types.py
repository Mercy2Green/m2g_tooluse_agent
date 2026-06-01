from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class VelocityLimits:
    """Clamp ranges for base velocity commands in the robot base frame."""

    vx: tuple[float, float] = (-1.0, 1.0)
    vy: tuple[float, float] = (-0.4, 0.4)
    wz: tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class VelocityCommand:
    """Base velocity command in the robot base frame."""

    vx: float
    vy: float
    wz: float
    duration_s: Optional[float] = None


@dataclass
class SkillResult:
    """Result returned by a navigation skill call."""

    success: bool
    message: str = ""
    elapsed_s: float = 0.0
    command: Optional[VelocityCommand] = None
    target: Optional[dict[str, float]] = field(default=None)


class BaseVelocityAdapter(Protocol):
    """Adapter protocol used by base navigation skills."""

    dt: float

    def set_velocity(
        self, vx: float, vy: float, wz: float, env_ids: Optional[Sequence[int]] = None
    ) -> None:
        """Set the desired base velocity command."""
        ...

    def stop(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Set the base velocity command to zero."""
        ...

    def step_for(self, duration_s: float) -> float:
        """Advance the environment for a duration and return elapsed simulated time."""
        ...
