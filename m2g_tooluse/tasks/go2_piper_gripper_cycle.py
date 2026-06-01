# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/Isaac-Lab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run the Go2 + Piper gripper cycle task."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go2 + Piper gripper cycle task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="m2g-go2-piper-simple-room-v0", help="Name of the task.")
parser.add_argument(
    "--switch_interval", type=int, default=60, help="Number of simulation steps before toggling the gripper state."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# make the m2g_tooluse package importable when running this file directly
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import m2g_tooluse.scene  # noqa: F401


def main():
    """Go2 + Piper gripper cycle agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # resolve the gripper joints once
    robot = env.unwrapped.scene["robot"]
    gripper_joint_ids, gripper_joint_names = robot.find_joints(["joint7", "joint8"], preserve_order=True)
    print(f"[INFO]: Gripper joints: {gripper_joint_names} -> {gripper_joint_ids}")

    # prepare actions
    open_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    close_actions = torch.zeros_like(open_actions)
    close_actions[:, gripper_joint_ids[0]] = -0.05
    close_actions[:, gripper_joint_ids[1]] = 0.05

    # reset environment
    env.reset()

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if (step_count // args_cli.switch_interval) % 2 == 0:
                actions = open_actions
            else:
                actions = close_actions
            env.step(actions)
            step_count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()