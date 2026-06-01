from __future__ import annotations

import math
from time import perf_counter
from typing import Optional

from .types import BaseVelocityAdapter, SkillResult, VelocityCommand, VelocityLimits


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


class Go2BaseNavSkill:
    """Time-integrated Go2 base navigation skill built on velocity commands.

    The public methods are intentionally pose-control shaped, but the first
    implementation computes durations from distance/rate. This keeps the API
    compatible with a later root-pose closed-loop implementation.
    """

    def __init__(self, adapter: BaseVelocityAdapter, limits: Optional[VelocityLimits] = None):
        self.adapter = adapter
        self.limits = limits or VelocityLimits()

    def walk_velocity(
        self, vx: float, vy: float, wz: float, duration_s: Optional[float] = None
    ) -> SkillResult:
        """Command base-frame velocity, optionally advancing the environment."""
        command = VelocityCommand(
            vx=_clamp(float(vx), self.limits.vx),
            vy=_clamp(float(vy), self.limits.vy),
            wz=_clamp(float(wz), self.limits.wz),
            duration_s=duration_s,
        )
        start = perf_counter()
        try:
            self.adapter.set_velocity(command.vx, command.vy, command.wz)
            elapsed_s = 0.0
            if duration_s is not None:
                elapsed_s = self.adapter.step_for(max(0.0, float(duration_s)))
            return SkillResult(
                success=True,
                message="velocity command applied",
                elapsed_s=elapsed_s or (perf_counter() - start),
                command=command,
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                message=str(exc),
                elapsed_s=perf_counter() - start,
                command=command,
            )

    def stop(self) -> SkillResult:
        """Stop the base by sending a zero velocity command."""
        start = perf_counter()
        try:
            self.adapter.stop()
            return SkillResult(
                success=True,
                message="stop command applied",
                elapsed_s=perf_counter() - start,
                command=VelocityCommand(0.0, 0.0, 0.0),
            )
        except Exception as exc:
            return SkillResult(success=False, message=str(exc), elapsed_s=perf_counter() - start)

    def move_forward(self, distance_m: float, speed_mps: float = 0.25) -> SkillResult:
        """Move forward by time-integrating a positive x velocity."""
        distance = abs(float(distance_m))
        speed = self._clamped_rate(abs(float(speed_mps)), self.limits.vx, positive=True)
        return self.walk_velocity(speed, 0.0, 0.0, self._duration(distance, speed, "speed_mps"))

    def move_backward(self, distance_m: float, speed_mps: float = 0.20) -> SkillResult:
        """Move backward by time-integrating a negative x velocity."""
        distance = abs(float(distance_m))
        speed = self._clamped_rate(abs(float(speed_mps)), self.limits.vx, positive=False)
        return self.walk_velocity(-speed, 0.0, 0.0, self._duration(distance, speed, "speed_mps"))

    def strafe_left(self, distance_m: float, speed_mps: float = 0.15) -> SkillResult:
        """Strafe left by time-integrating a positive y velocity."""
        distance = abs(float(distance_m))
        speed = self._clamped_rate(abs(float(speed_mps)), self.limits.vy, positive=True)
        return self.walk_velocity(0.0, speed, 0.0, self._duration(distance, speed, "speed_mps"))

    def strafe_right(self, distance_m: float, speed_mps: float = 0.15) -> SkillResult:
        """Strafe right by time-integrating a negative y velocity."""
        distance = abs(float(distance_m))
        speed = self._clamped_rate(abs(float(speed_mps)), self.limits.vy, positive=False)
        return self.walk_velocity(0.0, -speed, 0.0, self._duration(distance, speed, "speed_mps"))

    def turn_left(self, angle_deg: float, yaw_rate: float = 0.4) -> SkillResult:
        """Turn left by time-integrating a positive yaw rate."""
        return self.turn_yaw(math.radians(abs(float(angle_deg))), yaw_rate=yaw_rate)

    def turn_right(self, angle_deg: float, yaw_rate: float = 0.4) -> SkillResult:
        """Turn right by time-integrating a negative yaw rate."""
        return self.turn_yaw(-math.radians(abs(float(angle_deg))), yaw_rate=yaw_rate)

    def turn_yaw(self, angle_rad: float, yaw_rate: float = 0.4) -> SkillResult:
        """Turn by time-integrating yaw rate."""
        angle = float(angle_rad)
        rate = self._clamped_rate(abs(float(yaw_rate)), self.limits.wz, positive=angle >= 0.0)
        duration = self._duration(abs(angle), rate, "yaw_rate")
        signed_rate = math.copysign(rate, angle) if angle != 0.0 else 0.0
        result = self.walk_velocity(0.0, 0.0, signed_rate, duration)
        result.target = {"dyaw_rad": angle}
        return result

    def move_relative(
        self, dx_m: float = 0.0, dy_m: float = 0.0, dyaw_rad: float = 0.0
    ) -> SkillResult:
        """Execute relative x, y, and yaw motions sequentially."""
        start = perf_counter()
        results: list[SkillResult] = []
        if dx_m > 0.0:
            results.append(self.move_forward(dx_m))
        elif dx_m < 0.0:
            results.append(self.move_backward(abs(dx_m)))
        if dy_m > 0.0:
            results.append(self.strafe_left(dy_m))
        elif dy_m < 0.0:
            results.append(self.strafe_right(abs(dy_m)))
        if dyaw_rad:
            results.append(self.turn_yaw(dyaw_rad))

        failed = next((result for result in results if not result.success), None)
        elapsed_s = sum(result.elapsed_s for result in results) or (perf_counter() - start)
        if failed is not None:
            return SkillResult(
                success=False,
                message=f"move_relative failed: {failed.message}",
                elapsed_s=elapsed_s,
                target={"dx_m": dx_m, "dy_m": dy_m, "dyaw_rad": dyaw_rad},
            )
        return SkillResult(
            success=True,
            message="relative motion complete",
            elapsed_s=elapsed_s,
            target={"dx_m": dx_m, "dy_m": dy_m, "dyaw_rad": dyaw_rad},
        )

    @staticmethod
    def _duration(distance_or_angle: float, rate: float, rate_name: str) -> float:
        if rate <= 0.0:
            raise ValueError(f"{rate_name} must be positive")
        return abs(distance_or_angle) / rate

    @staticmethod
    def _clamped_rate(rate: float, bounds: tuple[float, float], positive: bool) -> float:
        bound = bounds[1] if positive else abs(bounds[0])
        clamped = min(abs(rate), abs(bound))
        if clamped <= 0.0:
            raise ValueError("requested direction is disabled by velocity limits")
        return clamped
