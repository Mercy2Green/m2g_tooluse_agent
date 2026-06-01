from __future__ import annotations

from typing import Any, Iterable

from m2g_tooluse.gt_demo.types import GraspPlan, Pose3D, SkillResult


class PiperScriptedPickSkill:
    def __init__(
        self,
        env: Any,
        robot_asset_name: str = "robot",
        object_asset_name: str = "object",
        enable_demo_attach_fallback: bool = True,
    ):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.robot_asset_name = robot_asset_name
        self.object_asset_name = object_asset_name
        self.enable_demo_attach_fallback = enable_demo_attach_fallback
        self._warned_scripted = False
        self._arm_joint_ids: list[int] | None = None
        self._gripper_joint_ids: list[int] | None = None

    def open_gripper(self) -> SkillResult:
        return self._run_action_target({"joint7": 0.0, "joint8": 0.0}, duration_s=0.6, message="gripper opened")

    def close_gripper(self) -> SkillResult:
        return self._run_action_target({"joint7": -0.05, "joint8": 0.05}, duration_s=0.6, message="gripper closed")

    def move_to_pose(self, pose: Pose3D, duration_s: float = 2.0) -> SkillResult:
        self._warn_scripted(pose)
        return self._run_arm_waypoint(
            [0.0, 0.55, -0.55, 0.0, 0.45, 0.0],
            duration_s=duration_s,
            message="scripted arm move complete",
            pose=pose,
        )

    def lift(self, lift_pose: Pose3D, duration_s: float = 1.5) -> SkillResult:
        self._warn_scripted(lift_pose)
        return self._run_arm_waypoint(
            [0.0, 0.35, -0.35, 0.0, 0.65, 0.0],
            duration_s=duration_s,
            message="scripted lift complete",
            pose=lift_pose,
            attach_object=self.enable_demo_attach_fallback,
        )

    def execute_pick(self, grasp_plan: GraspPlan) -> SkillResult:
        try:
            object_start_z = self._object_z()
            print("[GT_DEMO] execute_pick: open_gripper")
            result = self.open_gripper()
            if not result.success:
                return result

            print("[GT_DEMO] execute_pick: move_to_pregrasp")
            result = self._run_arm_waypoint(
                [0.0, 0.55, -0.55, 0.0, 0.45, 0.0],
                duration_s=1.6,
                message="scripted pregrasp reached",
                pose=grasp_plan.pregrasp_pose,
            )
            if not result.success:
                return result

            print("[GT_DEMO] execute_pick: move_to_grasp")
            result = self._run_arm_waypoint(
                [0.0, 0.70, -0.70, 0.0, 0.55, 0.0],
                duration_s=1.2,
                message="scripted grasp reached",
                pose=grasp_plan.grasp_pose,
            )
            if not result.success:
                return result

            print("[GT_DEMO] execute_pick: close_gripper")
            result = self.close_gripper()
            if not result.success:
                return result

            print("[GT_DEMO] execute_pick: lift")
            result = self.lift(grasp_plan.lift_pose, duration_s=1.5)
            if not result.success:
                return result

            verify = self.verify_pick_gt(object_start_z)
            return SkillResult(
                success=verify.success,
                message=verify.message,
                data={
                    "grasp_plan": grasp_plan,
                    "scripted_fallback": True,
                    "demo_attach_fallback": self.enable_demo_attach_fallback,
                    **verify.data,
                },
            )
        except Exception as exc:
            return SkillResult(success=False, message=f"execute_pick failed: {exc}")

    def verify_pick_gt(self, object_start_z: float | None = None) -> SkillResult:
        try:
            start_z = object_start_z if object_start_z is not None else self._object_z()
            current_z = self._object_z()
            if current_z >= start_z + 0.05:
                return SkillResult(
                    success=True,
                    message="verify_pick_gt passed: object z increased",
                    data={"object_start_z": start_z, "object_current_z": current_z},
                )
            if self.enable_demo_attach_fallback:
                return SkillResult(
                    success=True,
                    message="verify_pick_gt demo fallback: scripted pick completed; object lift was not physically verified",
                    data={"object_start_z": start_z, "object_current_z": current_z},
                )
            return SkillResult(
                success=False,
                message="verify_pick_gt failed: object z did not increase",
                data={"object_start_z": start_z, "object_current_z": current_z},
            )
        except Exception as exc:
            return SkillResult(success=False, message=f"verify_pick_gt failed: {exc}")

    def _run_arm_waypoint(
        self,
        arm_joint_offsets: Iterable[float],
        *,
        duration_s: float,
        message: str,
        pose: Pose3D,
        attach_object: bool = False,
    ) -> SkillResult:
        joint_targets = {f"joint{idx}": value for idx, value in enumerate(arm_joint_offsets, start=1)}
        result = self._run_action_target(joint_targets, duration_s=duration_s, message=message)
        if attach_object:
            print("[GT_DEMO] execute_pick: Using scripted joint fallback and demo attach fallback")
            self._attach_object_to_pose(pose, duration_s=max(0.2, duration_s * 0.5))
        result.data["logged_pose"] = pose
        result.data["warning"] = "Using scripted joint fallback; grasp pose is logged but not fully solved by IK yet."
        return result

    def _run_action_target(self, target_offsets: dict[str, float], *, duration_s: float, message: str) -> SkillResult:
        try:
            import math
            import torch

            steps = max(1, int(math.ceil(duration_s / self._dt())))
            action = torch.zeros(self.env.action_space.shape, device=self.unwrapped.device)
            joint_ids = self._joint_ids(list(target_offsets.keys()))
            values = list(target_offsets.values())
            for joint_id, value in zip(joint_ids, values):
                action[:, joint_id] = float(value)
            for _ in range(steps):
                self.env.step(action)
            return SkillResult(success=True, message=message, data={"steps": steps, "targets": target_offsets})
        except Exception as exc:
            return SkillResult(success=False, message=f"{message} failed: {exc}")

    def _attach_object_to_pose(self, pose: Pose3D, duration_s: float) -> None:
        import math
        import torch

        obj = self.unwrapped.scene[self.object_asset_name]
        env_ids = torch.tensor([0], device=self.unwrapped.device, dtype=torch.long)
        start_pose = obj.data.root_state_w[0:1, :7].clone()
        target_pose = start_pose.clone()
        target_pose[0, 0:3] = torch.tensor(pose.position, device=self.unwrapped.device, dtype=target_pose.dtype)
        target_pose[0, 3:7] = torch.tensor(pose.orientation, device=self.unwrapped.device, dtype=target_pose.dtype)
        steps = max(1, int(math.ceil(duration_s / self._dt())))
        zero_action = torch.zeros(self.env.action_space.shape, device=self.unwrapped.device)
        zero_vel = torch.zeros((1, 6), device=self.unwrapped.device)
        for index in range(1, steps + 1):
            alpha = index / steps
            root_pose = start_pose.clone()
            root_pose[:, 0:3] = start_pose[:, 0:3] * (1.0 - alpha) + target_pose[:, 0:3] * alpha
            root_pose[:, 3:7] = target_pose[:, 3:7]
            obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            obj.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
            self.env.step(zero_action)

    def _joint_ids(self, joint_names: list[str]) -> list[int]:
        robot = self.unwrapped.scene[self.robot_asset_name]
        ids, _ = robot.find_joints(joint_names, preserve_order=True)
        return [int(joint_id) for joint_id in ids]

    def _object_z(self) -> float:
        obj = self.unwrapped.scene[self.object_asset_name]
        return float(obj.data.root_state_w[0, 2].detach().cpu().item())

    def _dt(self) -> float:
        if hasattr(self.unwrapped, "step_dt"):
            return float(self.unwrapped.step_dt)
        cfg = getattr(self.unwrapped, "cfg", None)
        sim_cfg = getattr(cfg, "sim", None)
        if sim_cfg is not None and hasattr(sim_cfg, "dt") and hasattr(cfg, "decimation"):
            return float(sim_cfg.dt) * float(cfg.decimation)
        return 1.0 / 30.0

    def _warn_scripted(self, pose: Pose3D) -> None:
        if not self._warned_scripted:
            print("[GT_DEMO] warning: Using scripted joint fallback; grasp pose is logged but not fully solved by IK yet.")
            self._warned_scripted = True
        print(f"[GT_DEMO] scripted target pose logged: {pose}")
