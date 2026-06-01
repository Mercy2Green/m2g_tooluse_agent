from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from m2g_tooluse.gt_demo.types import SkillResult


class RslRlVelocityPolicyBaseController:
    """Run a trained Go2 leg velocity policy inside the IsaacLab process."""

    def __init__(
        self,
        env: Any,
        *,
        checkpoint_path: str | None = None,
        jit_path: str | None = None,
        task_id: str = "M2G-GT-Demo-Go2Piper-LocomotionPolicy-v0",
        robot_asset_name: str = "robot",
        clip_actions: float | None = None,
    ):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.task_id = task_id
        self.checkpoint_path = checkpoint_path
        self.jit_path = jit_path
        self.robot_asset_name = robot_asset_name
        self.device = self.unwrapped.device
        self.dt = self._compute_dt()
        self._policy: Any | None = None
        self._policy_nn: Any | None = None
        self._runner: Any | None = None
        self._wrapped_env: Any | None = None
        self._clip_actions = clip_actions

    def move_forward(
        self,
        duration_s: float = 3.0,
        vx: float = 0.25,
        yaw_rate: float = 0.0,
        min_distance: float = 0.10,
    ) -> SkillResult:
        try:
            import torch

            load_result = self._ensure_policy_loaded()
            if not load_result.success:
                return load_result

            robot = self.unwrapped.scene[self.robot_asset_name]
            start_pos = robot.data.root_pos_w[0].detach().clone()
            n_steps = max(1, int(math.ceil(max(0.0, duration_s) / self.dt)))
            last_action_shape: tuple[int, ...] | None = None
            last_obs_shape: Any = None
            done_count = 0
            self._set_base_velocity_command(vx, 0.0, yaw_rate)
            obs = self._get_observations()
            last_obs_shape = self._obs_shape(obs)

            for _ in range(n_steps):
                self._set_base_velocity_command(vx, 0.0, yaw_rate)
                with torch.inference_mode():
                    action = self._compute_action(obs)
                last_action_shape = tuple(action.shape)
                shape_result = self._validate_action_shape(action, obs)
                if not shape_result.success:
                    return shape_result
                self._set_base_velocity_command(vx, 0.0, yaw_rate)
                obs, _rew, dones, _extras = self._wrapped_env.step(action)
                if torch.any(dones):
                    done_count += int(torch.count_nonzero(dones).item())
                    self._reset_recurrent_policy(dones)

            self._set_base_velocity_command(vx, 0.0, yaw_rate)
            end_pos = robot.data.root_pos_w[0].detach().clone()
            delta = end_pos - start_pos
            distance_xy = float(torch.linalg.norm(delta[:2]).item())
            elapsed_s = n_steps * self.dt
            avg_speed_xy = distance_xy / elapsed_s if elapsed_s > 0.0 else 0.0
            success = distance_xy >= float(min_distance) and done_count == 0
            if success:
                message = "rl locomotion policy move_forward complete"
            elif done_count > 0:
                message = (
                    "rl locomotion policy move_forward did not complete cleanly: "
                    f"termination/reset observed {done_count} time(s)"
                )
            else:
                message = (
                    "rl locomotion policy move_forward did not move far enough: "
                    f"distance_xy={distance_xy:.3f} < min_distance={float(min_distance):.3f}"
                )
            return SkillResult(
                success=success,
                message=message,
                data={
                    "task_id": self.task_id,
                    "checkpoint_path": self.checkpoint_path,
                    "jit_path": self.jit_path,
                    "vx": float(vx),
                    "vy": 0.0,
                    "yaw_rate": float(yaw_rate),
                    "duration_s": float(duration_s),
                    "elapsed_s": float(elapsed_s),
                    "steps": int(n_steps),
                    "start_root_pos_w": self._to_list(start_pos),
                    "end_root_pos_w": self._to_list(end_pos),
                    "delta_root_pos_w": self._to_list(delta),
                    "distance_xy": distance_xy,
                    "min_distance": float(min_distance),
                    "avg_speed_xy": avg_speed_xy,
                    "done_count": done_count,
                    "obs_shape": last_obs_shape,
                    "policy_action_shape": last_action_shape,
                    "env_action_space": str(self._wrapped_env.action_space),
                },
            )
        except Exception as exc:
            return SkillResult(success=False, message=f"rl locomotion move_forward failed: {exc}", data=self._debug_data())

    def stop(self, duration_s: float = 1.0) -> SkillResult:
        try:
            import torch

            load_result = self._ensure_policy_loaded()
            if not load_result.success:
                return load_result
            n_steps = max(1, int(math.ceil(max(0.0, duration_s) / self.dt)))
            obs = self._get_observations()
            for _ in range(n_steps):
                self._set_base_velocity_command(0.0, 0.0, 0.0)
                with torch.inference_mode():
                    action = self._compute_action(obs)
                shape_result = self._validate_action_shape(action, obs)
                if not shape_result.success:
                    return shape_result
                obs, _rew, dones, _extras = self._wrapped_env.step(action)
                if torch.any(dones):
                    self._reset_recurrent_policy(dones)
            return SkillResult(success=True, message="rl locomotion policy stop complete", data={"steps": n_steps})
        except Exception as exc:
            return SkillResult(success=False, message=f"rl locomotion stop failed: {exc}", data=self._debug_data())

    def _ensure_policy_loaded(self) -> SkillResult:
        if self._policy is not None:
            return SkillResult(success=True, message="policy already loaded")
        if self.checkpoint_path:
            path = Path(self.checkpoint_path).expanduser()
            if not path.exists():
                return SkillResult(
                    success=False,
                    message=(
                        "RSL-RL checkpoint path does not exist. Set "
                        "M2G_GO2PIPER_POLICY_CKPT or pass --policy-checkpoint."
                    ),
                    data={"checkpoint_path": str(path), "task_id": self.task_id},
                )
            return self._load_rsl_rl_checkpoint(str(path))
        if self.jit_path:
            path = Path(self.jit_path).expanduser()
            if not path.exists():
                return SkillResult(
                    success=False,
                    message="TorchScript policy path does not exist. Set M2G_GO2PIPER_POLICY_JIT or pass --policy-jit.",
                    data={"jit_path": str(path), "task_id": self.task_id},
                )
            return self._load_jit_policy(str(path))
        return SkillResult(
            success=False,
            message=(
                "No locomotion policy provided. Set M2G_GO2PIPER_POLICY_CKPT=/path/to/checkpoint.pt "
                "or pass --policy-checkpoint /path/to/checkpoint.pt."
            ),
            data={"task_id": self.task_id},
        )

    def _load_rsl_rl_checkpoint(self, checkpoint_path: str) -> SkillResult:
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg = load_cfg_from_registry(self.task_id, "rsl_rl_cfg_entry_point")
            self._wrapped_env = RslRlVecEnvWrapper(self.env, clip_actions=self._clip_actions or agent_cfg.clip_actions)
            runner = OnPolicyRunner(self._wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint_path)
            self._runner = runner
            self._policy = runner.get_inference_policy(device=self.unwrapped.device)
            self._policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
            return SkillResult(success=True, message=f"loaded RSL-RL checkpoint: {checkpoint_path}")
        except Exception as exc:
            return SkillResult(
                success=False,
                message=f"failed to load RSL-RL checkpoint: {exc}",
                data={**self._debug_data(), "checkpoint_path": checkpoint_path},
            )

    def _load_jit_policy(self, jit_path: str) -> SkillResult:
        try:
            import torch
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

            self._wrapped_env = RslRlVecEnvWrapper(self.env, clip_actions=self._clip_actions)
            self._policy = torch.jit.load(jit_path, map_location=self.unwrapped.device)
            self._policy.eval()
            return SkillResult(success=True, message=f"loaded TorchScript policy: {jit_path}")
        except Exception as exc:
            return SkillResult(
                success=False,
                message=f"failed to load TorchScript policy: {exc}",
                data={**self._debug_data(), "jit_path": jit_path},
            )

    def _get_observations(self) -> Any:
        return self._wrapped_env.get_observations()

    def _compute_action(self, obs: Any) -> Any:
        if self.jit_path and not self.checkpoint_path:
            policy_obs = obs["policy"] if isinstance(obs, dict) or hasattr(obs, "__getitem__") else obs
            return self._policy(policy_obs)
        return self._policy(obs)

    def _validate_action_shape(self, action: Any, obs: Any) -> SkillResult:
        expected_dim = int(getattr(self._wrapped_env, "num_actions", -1))
        actual_dim = int(action.shape[-1]) if hasattr(action, "shape") and len(action.shape) > 0 else -1
        if actual_dim == expected_dim:
            return SkillResult(success=True, message="action shape ok")
        return SkillResult(
            success=False,
            message=(
                "policy action dimension does not match env action dimension; "
                f"expected {expected_dim}, got {actual_dim}"
            ),
            data={
                "task_id": self.task_id,
                "checkpoint_path": self.checkpoint_path,
                "jit_path": self.jit_path,
                "env_action_space": str(getattr(self._wrapped_env, "action_space", None)),
                "policy_action_shape": tuple(action.shape) if hasattr(action, "shape") else str(type(action)),
                "obs_shape": self._obs_shape(obs),
            },
        )

    def _set_base_velocity_command(self, vx: float, vy: float, yaw_rate: float) -> None:
        import torch

        command_manager = self.unwrapped.command_manager
        command = command_manager.get_command("base_velocity")
        target = torch.tensor([[vx, vy, yaw_rate]], device=command.device, dtype=command.dtype)
        command[:] = target.expand_as(command)
        term = command_manager.get_term("base_velocity")
        # UniformVelocityCommand exposes command as a writable view of
        # vel_command_b in current IsaacLab. We also clear standing flags so the
        # term's next _update_command() call does not zero our forced command.
        if hasattr(term, "vel_command_b"):
            term.vel_command_b[:] = command
        if hasattr(term, "is_standing_env"):
            term.is_standing_env[:] = False
        if hasattr(term, "is_heading_env"):
            term.is_heading_env[:] = False

    def _reset_recurrent_policy(self, dones: Any) -> None:
        if self._policy_nn is not None and hasattr(self._policy_nn, "reset"):
            self._policy_nn.reset(dones)
        elif self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def _compute_dt(self) -> float:
        if hasattr(self.unwrapped, "step_dt"):
            return float(self.unwrapped.step_dt)
        cfg = getattr(self.unwrapped, "cfg", None)
        sim_cfg = getattr(cfg, "sim", None)
        if sim_cfg is not None and hasattr(sim_cfg, "dt") and hasattr(cfg, "decimation"):
            return float(sim_cfg.dt) * float(cfg.decimation)
        return 1.0 / 30.0

    def _obs_shape(self, obs: Any) -> Any:
        if isinstance(obs, dict):
            return {key: tuple(value.shape) if hasattr(value, "shape") else str(type(value)) for key, value in obs.items()}
        if hasattr(obs, "keys"):
            return {key: tuple(obs[key].shape) if hasattr(obs[key], "shape") else str(type(obs[key])) for key in obs.keys()}
        return tuple(obs.shape) if hasattr(obs, "shape") else str(type(obs))

    def _debug_data(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "checkpoint_path": self.checkpoint_path,
            "jit_path": self.jit_path,
            "env_action_space": str(getattr(self.env, "action_space", None)),
            "wrapped_action_space": str(getattr(self._wrapped_env, "action_space", None)),
        }

    @staticmethod
    def _to_list(tensor: Any) -> list[float]:
        return [float(value) for value in tensor.detach().cpu().tolist()]
