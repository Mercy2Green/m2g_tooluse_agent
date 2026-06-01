from __future__ import annotations

from typing import Any, Callable

from m2g_tooluse.gt_demo.skill.base_rl_policy import RslRlVelocityPolicyBaseController
from m2g_tooluse.gt_demo.types import SkillResult


class GtDemoLocomotionController:
    """Facade for the RL locomotion-policy gt_demo task."""

    def __init__(
        self,
        env: Any,
        *,
        checkpoint_path: str | None = None,
        jit_path: str | None = None,
        task_id: str = "M2G-GT-Demo-Go2Piper-LocomotionPolicy-v0",
        move_duration: float = 3.0,
        move_vx: float = 0.25,
        move_yaw_rate: float = 0.0,
        min_distance: float = 0.10,
    ):
        self.env = env
        self.task_id = task_id
        self.move_duration = float(move_duration)
        self.move_vx = float(move_vx)
        self.move_yaw_rate = float(move_yaw_rate)
        self.min_distance = float(min_distance)
        self.base_rl = RslRlVelocityPolicyBaseController(
            env,
            checkpoint_path=checkpoint_path,
            jit_path=jit_path,
            task_id=task_id,
        )
        self.last_result = SkillResult(success=True, message="gt_demo locomotion controller initialized")
        self.last_command = "idle"

    def start(self) -> SkillResult:
        self.last_command = "start"
        self.last_result = SkillResult(success=True, message="gt_demo locomotion policy controller is ready")
        return self.last_result

    def reset(self) -> SkillResult:
        self.last_command = "reset"
        self.env.reset()
        self.last_result = SkillResult(success=True, message="gt_demo locomotion env reset complete")
        return self.last_result

    def move_forward_policy(self) -> SkillResult:
        self.last_command = "move_forward_policy"
        self.last_result = self.base_rl.move_forward(
            duration_s=self.move_duration,
            vx=self.move_vx,
            yaw_rate=self.move_yaw_rate,
            min_distance=self.min_distance,
        )
        return self.last_result

    def stop_policy(self) -> SkillResult:
        self.last_command = "stop_policy"
        self.last_result = self.base_rl.stop(duration_s=1.0)
        return self.last_result

    def run_locomotion_test(self) -> SkillResult:
        self.last_command = "run_locomotion_test"
        move_result = self.base_rl.move_forward(
            duration_s=self.move_duration,
            vx=self.move_vx,
            yaw_rate=self.move_yaw_rate,
            min_distance=self.min_distance,
        )
        if not move_result.success:
            self.last_result = move_result
            return move_result
        stop_result = self.base_rl.stop(duration_s=0.5)
        self.last_result = SkillResult(
            success=move_result.success and stop_result.success,
            message=f"{move_result.message}; {stop_result.message}",
            data={"move": move_result.data, "stop": stop_result.data},
        )
        return self.last_result

    def status(self) -> SkillResult:
        return SkillResult(
            success=self.last_result.success,
            message=f"last_command={self.last_command}; last_result={self.last_result.message}",
            data=self.last_result.data,
        )

    def dispatch(self, command: str) -> SkillResult:
        handlers: dict[str, Callable[[], SkillResult]] = {
            "start": self.start,
            "reset": self.reset,
            "move_forward_policy": self.move_forward_policy,
            "stop_policy": self.stop_policy,
            "run_locomotion_test": self.run_locomotion_test,
            "status": self.status,
        }
        handler = handlers.get(command)
        if handler is None:
            return SkillResult(success=False, message=f"unknown gt_demo locomotion command: {command}")
        return handler()
