from __future__ import annotations

"""Inspect and smoke-test the Go2+Piper locomotion training env wiring."""

import argparse
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_TASK = "M2G-Navigation-Go2Piper-Velocity-Flat-Fixed-v0"

parser = argparse.ArgumentParser(description="Inspect Go2+Piper velocity locomotion env.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--zero-action-seconds", type=float, default=10.0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--gui", action="store_true", help="Force GUI mode for visual inspection.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.gui:
    args_cli.headless = False

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import m2g_tooluse.train.navigation  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


PIPER_JOINTS = {f"joint{i}" for i in range(1, 9)}
GO2_LEG_JOINTS = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
GO2_FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
GO2_THIGH_BODIES = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]


def _maybe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    try:
        return [str(item) for item in list(value)]
    except TypeError:
        return [str(value)]


def _term_joint_names(action_term: Any) -> list[str]:
    for attr in ("joint_names", "_joint_names"):
        names = getattr(action_term, attr, None)
        if names is not None:
            return _maybe_list(names)
    joint_ids = getattr(action_term, "_joint_ids", None)
    asset = getattr(action_term, "_asset", None)
    if joint_ids is not None and asset is not None:
        robot_joint_names = _maybe_list(getattr(asset, "joint_names", []))
        return [robot_joint_names[int(idx)] for idx in joint_ids]
    return []


def _sensor_body_names(term_cfg: Any) -> Any:
    if term_cfg is None:
        return None
    return term_cfg.params.get("sensor_cfg").body_names


def _joint_asset_names(term_cfg: Any) -> Any:
    if term_cfg is None:
        return None
    asset_cfg = term_cfg.params.get("asset_cfg") if term_cfg.params else None
    return None if asset_cfg is None else asset_cfg.joint_names


def _print_cfg_wiring(env_cfg: Any) -> None:
    print(f"contact_forces prim_path: {env_cfg.scene.contact_forces.prim_path}", flush=True)
    print(
        "base_contact sensor_cfg body_names: "
        f"{_sensor_body_names(getattr(env_cfg.terminations, 'base_contact', None))}",
        flush=True,
    )
    print(
        f"feet_air_time sensor_cfg body_names: {_sensor_body_names(getattr(env_cfg.rewards, 'feet_air_time', None))}",
        flush=True,
    )
    print(
        "undesired_contacts sensor_cfg body_names: "
        f"{_sensor_body_names(getattr(env_cfg.rewards, 'undesired_contacts', None))}",
        flush=True,
    )
    print(
        "observation joint_pos asset_cfg joint_names: "
        f"{_joint_asset_names(env_cfg.observations.policy.joint_pos)}",
        flush=True,
    )
    print(
        "observation joint_vel asset_cfg joint_names: "
        f"{_joint_asset_names(env_cfg.observations.policy.joint_vel)}",
        flush=True,
    )
    for reward_name in ("dof_torques_l2", "dof_acc_l2", "dof_pos_limits"):
        print(
            f"reward {reward_name} asset_cfg joint_names: "
            f"{_joint_asset_names(getattr(env_cfg.rewards, reward_name, None))}",
            flush=True,
        )


def _piper_link_heights(robot: Any, body_names: list[str]) -> dict[str, float]:
    heights = {}
    body_pos_w = robot.data.body_pos_w[0]
    for idx, name in enumerate(body_names):
        if name in {"arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "camera_link", "link7", "link8"}:
            heights[name] = float(body_pos_w[idx, 2].detach().cpu())
    return heights


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _print_cfg_wiring(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        print(f"task id: {args_cli.task}", flush=True)
        print(f"observation space: {env.observation_space}", flush=True)
        print(f"action space: {env.action_space}", flush=True)

        robot = env.unwrapped.scene["robot"]
        joint_names = _maybe_list(getattr(robot, "joint_names", []))
        body_names = _maybe_list(getattr(robot, "body_names", []))
        print(f"robot joint_names ({len(joint_names)}): {joint_names}", flush=True)
        print(f"robot body names ({len(body_names)}): {body_names}", flush=True)

        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        selected_joint_names = _term_joint_names(action_term)
        action_dim = env.action_space.shape[-1]
        print(f"action dimension: {action_dim}", flush=True)
        print(f"selected action joint names: {selected_joint_names}", flush=True)
        print(f"selected action joint count: {len(selected_joint_names)}", flush=True)

        if action_dim != 12:
            print(f"ERROR: expected action dimension 12, got {action_dim}", flush=True)
            raise SystemExit(2)
        piper_action_joints = [name for name in selected_joint_names if name in PIPER_JOINTS]
        if piper_action_joints:
            print(f"ERROR: Piper joints are in policy action space: {piper_action_joints}", flush=True)
            raise SystemExit(2)

        print(f'exact base body match result for "base": {[name for name in body_names if name == "base"]}', flush=True)
        print(f"foot body match result: {[name for name in body_names if name in GO2_FOOT_BODIES]}", flush=True)
        print(f"thigh body match result: {[name for name in body_names if name in GO2_THIGH_BODIES]}", flush=True)

        contact_sensor = env.unwrapped.scene.sensors.get("contact_forces")
        if contact_sensor is not None:
            print(f"resolved contact sensor body_names: {contact_sensor.body_names}", flush=True)

        command_manager = getattr(env.unwrapped, "command_manager", None)
        command_terms = list(getattr(command_manager, "_terms", {}).keys()) if command_manager is not None else []
        print(f"command manager terms: {command_terms}", flush=True)
        if command_manager is not None:
            command = command_manager.get_command("base_velocity")
            print(f"base_velocity command tensor shape: {tuple(command.shape)}", flush=True)

        env.reset()
        root_height = float(robot.data.root_pos_w[0, 2].detach().cpu())
        print(f"current root height after reset: {root_height}", flush=True)
        print(f"piper link heights after reset: {_piper_link_heights(robot, body_names)}", flush=True)

        step_dt = env.unwrapped.step_dt
        steps = max(1, int(args_cli.zero_action_seconds / step_dt))
        zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        root_heights: list[float] = []
        max_base_contact_force = 0.0
        base_contact_triggered = False
        time_out_triggered = False
        first_done_step: int | None = None

        base_contact_enabled = "base_contact" in getattr(env.unwrapped.termination_manager, "_term_names", [])
        base_sensor_index = None
        if contact_sensor is not None and "base" in contact_sensor.body_names:
            base_sensor_index = contact_sensor.body_names.index("base")

        for step_idx in range(steps):
            _, _, terminated, truncated, _ = env.step(zero_action)
            height = float(robot.data.root_pos_w[0, 2].detach().cpu())
            root_heights.append(height)

            if contact_sensor is not None and base_sensor_index is not None:
                force = contact_sensor.data.net_forces_w[0, base_sensor_index].norm().detach().cpu().item()
                max_base_contact_force = max(max_base_contact_force, float(force))

            if base_contact_enabled:
                base_term = env.unwrapped.termination_manager.get_term("base_contact")
                base_contact_triggered = base_contact_triggered or bool(base_term[0].detach().cpu().item())
            time_out_triggered = time_out_triggered or bool(truncated[0].detach().cpu().item())
            done_now = bool((terminated[0] | truncated[0]).detach().cpu().item())
            if done_now and first_done_step is None:
                first_done_step = step_idx
                break

        print(
            "zero-action standing test: "
            f"seconds={args_cli.zero_action_seconds}, steps={steps}, "
            f"min_root_height={min(root_heights)}, max_root_height={max(root_heights)}, "
            f"max_base_contact_force={max_base_contact_force}, "
            f"base_contact_triggered={base_contact_triggered}, "
            f"time_out_triggered={time_out_triggered}, first_done_step={first_done_step}",
            flush=True,
        )

        if base_contact_triggered:
            print("ZERO_ACTION_STAND_FAILED: fix asset/init/contact config before training.", flush=True)
            raise SystemExit(3)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
