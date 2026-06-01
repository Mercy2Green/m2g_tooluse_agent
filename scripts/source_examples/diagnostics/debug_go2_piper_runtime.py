#!/usr/bin/env python3
"""Runtime probes for the Go2+Piper velocity env.

Run through IsaacLab/IsaacSim, for example:

    cd $M2G_TOOLUSE_ROOT
    conda activate isaacsim5
    $ISAACLAB_PATH/isaaclab.sh -p scripts/source_examples/diagnostics/debug_go2_piper_runtime.py \
      --task M2G-Navigation-Go2Piper-Velocity-Flat-Play-v0 --num_envs 4 --steps 64 --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


THIS_FILE = Path(__file__).resolve()
M2G_ROOT = THIS_FILE.parents[3]
WORKSPACE_ROOT = M2G_ROOT.parent

sys.path.insert(0, str(M2G_ROOT))

parser = argparse.ArgumentParser(description="Print Go2+Piper runtime velocity/torque diagnostics.")
parser.add_argument("--task", default="M2G-Navigation-Go2Piper-Velocity-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--base_body_name", default="base")
parser.add_argument("--nonzero_action", action="store_true", help="Use small random actions instead of zero actions.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import m2g_tooluse.train.navigation  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x_abs = x.detach().abs()
    return {"mean_abs": float(x_abs.mean().cpu()), "max_abs": float(x_abs.max().cpu())}


def norm_xy(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x[..., :2], dim=-1)


def resolved_action_joint_names(env) -> list[str]:
    term = getattr(env.action_manager, "_terms", {}).get("joint_pos")
    if term is None:
        return []
    return list(getattr(term, "_joint_names", []) or [])


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    env.reset()
    action_shape = env.action_space.shape
    action_dim = action_shape[-1] if action_shape else unwrapped.action_manager.total_action_dim

    base_ids, base_names = robot.find_bodies(args_cli.base_body_name)
    if len(base_ids) == 0:
        base_ids, base_names = robot.find_bodies(".*base.*")
    if len(base_ids) == 0:
        raise RuntimeError(f"Could not resolve base body from {args_cli.base_body_name!r}; bodies={robot.body_names}")
    base_id = int(base_ids[0])
    base_name = str(base_names[0])

    print("## Resolved Names")
    print(f"task: {args_cli.task}")
    print(f"num_envs: {unwrapped.num_envs}")
    print(f"action_dim: {action_dim}")
    print(f"resolved action joint names: {resolved_action_joint_names(unwrapped)}")
    print(f"resolved base body id/name: {base_id} / {base_name}")
    print(f"all body names: {robot.body_names}")

    device = unwrapped.device
    for step in range(args_cli.steps):
        if args_cli.nonzero_action:
            actions = 0.05 * torch.randn((unwrapped.num_envs, action_dim), device=device)
        else:
            actions = torch.zeros((unwrapped.num_envs, action_dim), device=device)
        obs, rewards, terminated, truncated, info = env.step(actions)

        if step in (0, args_cli.steps - 1):
            command = unwrapped.command_manager.get_command("base_velocity")
            root_xy_vel = norm_xy(robot.data.root_lin_vel_w)
            base_xy_vel = norm_xy(robot.data.body_lin_vel_w[:, base_id, :])
            root_z = robot.data.root_pos_w[:, 2]
            base_z = robot.data.body_pos_w[:, base_id, 2]
            print(f"\n## Step {step}")
            print(f"command xy norm mean/max: {float(norm_xy(command).mean().cpu()):.4f} / {float(norm_xy(command).max().cpu()):.4f}")
            print(f"root xy velocity norm mean/max: {float(root_xy_vel.mean().cpu()):.4f} / {float(root_xy_vel.max().cpu()):.4f}")
            print(f"Go2 base body xy velocity norm mean/max: {float(base_xy_vel.mean().cpu()):.4f} / {float(base_xy_vel.max().cpu()):.4f}")
            print(f"root_z mean/min: {float(root_z.mean().cpu()):.4f} / {float(root_z.min().cpu()):.4f}")
            print(f"Go2 base body z mean/min: {float(base_z.mean().cpu()):.4f} / {float(base_z.min().cpu()):.4f}")
            print(f"applied_torque mean/max abs: {tensor_stats(robot.data.applied_torque)}")
            print(f"computed_torque mean/max abs: {tensor_stats(robot.data.computed_torque)}")
            print(f"joint_vel mean/max abs: {tensor_stats(robot.data.joint_vel)}")

    print("\n## Interpretation Checklist")
    print("root and base velocity close: robot.data.root_* likely corresponds to Go2 base.")
    print("command nonzero but root/base velocity small: robot is physically not moving.")
    print("computed_torque high but applied_torque clipped: torque/effort limit is active.")
    print("joint_vel near a configured limit, especially calf: velocity_limit may be restricting gait.")
    print("torque and joint_vel modest while not moving: inspect reward/curriculum/action scale/policy behavior.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
