from __future__ import annotations

"""Run the Go2+Piper RL locomotion-policy gt_demo test."""

import sys

_SCRIPT_DIR = __file__.rsplit("/", 1)[0]
_WORKSPACE_ROOT = __file__.rsplit("/", 3)[0]
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import argparse
import os
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_TASK = "M2G-GT-Demo-Go2Piper-LocomotionPolicy-v0"

parser = argparse.ArgumentParser(description="Run the m2g gt_demo RL locomotion-policy test.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--policy-checkpoint", type=str, default=os.environ.get("M2G_GO2PIPER_POLICY_CKPT"))
parser.add_argument("--policy-jit", type=str, default=os.environ.get("M2G_GO2PIPER_POLICY_JIT"))
parser.add_argument("--object-name", type=str, default="object", help="Reserved; locomotion test does not use it.")
parser.add_argument("--ros2-bridge", action="store_true", default=False, help="Run local RPC server for ROS2 bridge.")
parser.add_argument("--no-auto-ros2-node", action="store_true", default=False)
parser.add_argument("--ros2-python", type=str, default="/usr/bin/python3")
parser.add_argument("--rpc-host", type=str, default="127.0.0.1")
parser.add_argument("--rpc-port", type=int, default=8765)
parser.add_argument("--move-duration", type=float, default=3.0)
parser.add_argument("--move-vx", type=float, default=0.25)
parser.add_argument("--move-yaw-rate", type=float, default=0.0)
parser.add_argument("--min-distance", type=float, default=0.10)
parser.add_argument("--keep-open", action="store_true", default=False)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.policy_checkpoint and not args_cli.policy_jit:
    print(
        "[GT_DEMO][LOCO] ERROR: no policy provided. Set:\n"
        "  export M2G_GO2PIPER_POLICY_CKPT=/path/to/checkpoint.pt\n"
        "or pass:\n"
        "  --policy-checkpoint /path/to/checkpoint.pt\n"
        "TorchScript is also supported with M2G_GO2PIPER_POLICY_JIT or --policy-jit.",
        file=sys.stderr,
    )
    sys.exit(2)

WORKSPACE_ROOT = Path(_WORKSPACE_ROOT)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import m2g_tooluse.train.navigation  # noqa: F401
import m2g_tooluse.gt_demo.task.registry  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from m2g_tooluse.gt_demo.locomotion_controller import GtDemoLocomotionController
from m2g_tooluse.gt_demo.ros2_bridge.rpc_server import LocalGtDemoRpcServer


def _make_controller(env) -> GtDemoLocomotionController:
    return GtDemoLocomotionController(
        env,
        checkpoint_path=args_cli.policy_checkpoint,
        jit_path=args_cli.policy_jit,
        task_id=args_cli.task,
        move_duration=args_cli.move_duration,
        move_vx=args_cli.move_vx,
        move_yaw_rate=args_cli.move_yaw_rate,
        min_distance=args_cli.min_distance,
    )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print(f"[GT_DEMO][LOCO] task={args_cli.task}")
        print(f"[GT_DEMO][LOCO] policy_checkpoint={args_cli.policy_checkpoint}")
        print(f"[GT_DEMO][LOCO] policy_jit={args_cli.policy_jit}")
        print(f"[GT_DEMO][LOCO] observation_space={env.observation_space}")
        print(f"[GT_DEMO][LOCO] action_space={env.action_space}")
        env.reset()

        if args_cli.ros2_bridge:
            controller = _make_controller(env)
            rpc_server = LocalGtDemoRpcServer(host=args_cli.rpc_host, port=args_cli.rpc_port)
            ros2_node_process = None
            rpc_server.start()
            if not args_cli.no_auto_ros2_node:
                node_script = WORKSPACE_ROOT / "m2g_tooluse" / "gt_demo" / "ros2_bridge" / "gt_demo_node.py"
                ros2_node_process = subprocess.Popen(
                    [
                        args_cli.ros2_python,
                        str(node_script),
                        "--rpc-host",
                        args_cli.rpc_host,
                        "--rpc-port",
                        str(args_cli.rpc_port),
                        "--service-profile",
                        "locomotion",
                    ],
                    env=os.environ.copy(),
                )
                print(f"[GT_DEMO][LOCO][ROS2] bridge node started with pid={ros2_node_process.pid}")
            print("[GT_DEMO][LOCO] ros2 bridge mode enabled; waiting for ROS2 service commands")
            zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            try:
                while simulation_app.is_running():
                    command = rpc_server.pop_next_command()
                    if command is None:
                        env.step(zero_action)
                        continue
                    print(f"[GT_DEMO][LOCO][RPC] executing command: {command}")
                    result = controller.dispatch(command)
                    print(f"[GT_DEMO][LOCO][RPC] result: {result}")
                    rpc_server.finish_command(result)
            finally:
                if ros2_node_process is not None:
                    ros2_node_process.terminate()
                    try:
                        ros2_node_process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        ros2_node_process.kill()
                rpc_server.close()
        else:
            controller = _make_controller(env)
            result = controller.run_locomotion_test()
            print(f"[GT_DEMO][LOCO] final SkillResult: {result}")

        if args_cli.keep_open and not args_cli.headless:
            print("[GT_DEMO][LOCO] keep-open enabled; stepping simulation until window closes")
            zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            while simulation_app.is_running():
                env.step(zero_action)
    except Exception as exc:
        print(f"[GT_DEMO][LOCO] ERROR: {exc}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
