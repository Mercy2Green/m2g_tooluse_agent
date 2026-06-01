from __future__ import annotations

import math
from typing import Any, Optional

from m2g_tooluse.gt_demo.types import ObjectInfo, SkillResult


class KinematicDebugBaseController:
    """Kinematic base controller for the gt_demo staged path.

    This controller is for gt_demo only and intentionally bypasses physical
    quadruped locomotion.
    """

    def __init__(self, env: Any, robot_asset_name: str = "robot"):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.robot_asset_name = robot_asset_name
        self.dt = self._compute_dt()

    def move_base_to_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        duration_s: float = 3.0,
        steps: Optional[int] = None,
    ) -> SkillResult:
        try:
            import torch

            robot = self.unwrapped.scene[self.robot_asset_name]
            env_ids = torch.tensor([0], device=self.unwrapped.device, dtype=torch.long)
            start_state = robot.data.root_state_w[0:1].clone()
            start_pose = start_state[:, :7].clone()
            target_pose = start_pose.clone()
            target_pose[0, 0] = float(x)
            target_pose[0, 1] = float(y)
            target_pose[0, 3:7] = torch.tensor(
                self._quat_from_yaw(float(yaw)),
                device=target_pose.device,
                dtype=target_pose.dtype,
            )
            n_steps = steps or max(1, int(math.ceil(max(0.0, duration_s) / self.dt)))
            for index in range(1, n_steps + 1):
                alpha = index / n_steps
                pose = start_pose.clone()
                pose[:, 0:3] = start_pose[:, 0:3] * (1.0 - alpha) + target_pose[:, 0:3] * alpha
                pose[:, 3:7] = target_pose[:, 3:7]
                robot.write_root_pose_to_sim(pose, env_ids=env_ids)
                robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.unwrapped.device), env_ids=env_ids)
                self._step_zero_action()
            return SkillResult(
                success=True,
                message="kinematic base pose updated",
                data={"x": float(x), "y": float(y), "yaw": float(yaw), "steps": n_steps},
            )
        except Exception as exc:
            return SkillResult(success=False, message=f"move_base_to_pose failed: {exc}")

    def move_base_near_object(
        self,
        object_info: ObjectInfo,
        standoff_distance: float = 0.8,
        duration_s: float = 3.0,
    ) -> SkillResult:
        ox, oy, _ = object_info.pose.position
        # Simple safe convention for MVP: stand on negative world-x side and face +x.
        base_x = ox - float(standoff_distance)
        base_y = oy
        yaw = 0.0
        return self.move_base_to_pose(base_x, base_y, yaw, duration_s=duration_s)

    def stop(self) -> SkillResult:
        try:
            robot = self.unwrapped.scene[self.robot_asset_name]
            robot.write_root_velocity_to_sim(self._zeros(1, 6), env_ids=self._env_ids())
            self._step_zero_action()
            return SkillResult(success=True, message="kinematic base stopped")
        except Exception as exc:
            return SkillResult(success=False, message=f"stop failed: {exc}")

    def _compute_dt(self) -> float:
        if hasattr(self.unwrapped, "step_dt"):
            return float(self.unwrapped.step_dt)
        cfg = getattr(self.unwrapped, "cfg", None)
        sim_cfg = getattr(cfg, "sim", None)
        if sim_cfg is not None and hasattr(sim_cfg, "dt") and hasattr(cfg, "decimation"):
            return float(sim_cfg.dt) * float(cfg.decimation)
        return 1.0 / 30.0

    def _step_zero_action(self) -> None:
        action = self._zero_action()
        self.env.step(action)

    def _zero_action(self) -> Any:
        import torch

        return torch.zeros(self.env.action_space.shape, device=self.unwrapped.device)

    def _env_ids(self) -> Any:
        import torch

        return torch.tensor([0], device=self.unwrapped.device, dtype=torch.long)

    def _zeros(self, rows: int, cols: int) -> Any:
        import torch

        return torch.zeros((rows, cols), device=self.unwrapped.device)

    @staticmethod
    def _quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
        half = yaw * 0.5
        return (math.cos(half), 0.0, 0.0, math.sin(half))
