from __future__ import annotations

"""Run the isolated Go2+Piper gt_demo staged pick flow."""

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

parser = argparse.ArgumentParser(description="Run the isolated m2g_tooluse gt_demo.")
parser.add_argument("--task", type=str, default="M2G-GT-Demo-Go2Piper-SimpleRoom-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--object-name", type=str, default="object")
parser.add_argument("--no-demo-attach-fallback", action="store_true", default=False)
parser.add_argument("--pause-before-start", action="store_true", default=False)
parser.add_argument("--keep-open", action="store_true", default=False)
parser.add_argument("--ros2-bridge", action="store_true", default=False, help="Run local RPC server for ROS2 bridge.")
parser.add_argument("--no-auto-ros2-node", action="store_true", default=False)
parser.add_argument("--ros2-python", type=str, default="/usr/bin/python3")
parser.add_argument("--rpc-host", type=str, default="127.0.0.1")
parser.add_argument("--rpc-port", type=int, default=8765)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

WORKSPACE_ROOT = Path(_WORKSPACE_ROOT)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import m2g_tooluse.gt_demo.task.registry  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from m2g_tooluse.gt_demo.controller import GtDemoController
from m2g_tooluse.gt_demo.ros2_bridge.rpc_server import LocalGtDemoRpcServer
from m2g_tooluse.gt_demo.search_and_pick import GtDemoSearchAndPick


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print(f"[GT_DEMO] task={args_cli.task}")
        print(f"[GT_DEMO] observation_space={env.observation_space}")
        print(f"[GT_DEMO] action_space={env.action_space}")
        env.reset()
        if args_cli.pause_before_start:
            input("[GT_DEMO] Press Enter to start search_and_pick...")

        if args_cli.ros2_bridge:
            controller = GtDemoController(
                env,
                object_name=args_cli.object_name,
                enable_demo_attach_fallback=not args_cli.no_demo_attach_fallback,
            )
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
                        "gt",
                    ],
                    env=os.environ.copy(),
                )
                print(f"[GT_DEMO][ROS2] bridge node started with pid={ros2_node_process.pid}")
            print("[GT_DEMO] ros2 bridge mode enabled; waiting for ROS2 service commands")
            zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            try:
                while simulation_app.is_running():
                    command = rpc_server.pop_next_command()
                    if command is None:
                        env.step(zero_action)
                        continue
                    print(f"[GT_DEMO][RPC] executing command: {command}")
                    result = controller.dispatch(command)
                    print(f"[GT_DEMO][RPC] result: {result}")
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
            demo = GtDemoSearchAndPick(
                env,
                enable_demo_attach_fallback=not args_cli.no_demo_attach_fallback,
            )
            result = demo.search_and_pick(args_cli.object_name)
            print(f"[GT_DEMO] final SkillResult: {result}")

        if not args_cli.ros2_bridge and args_cli.keep_open and not args_cli.headless:
            print("[GT_DEMO] keep-open enabled; stepping simulation until window closes")
            zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            while simulation_app.is_running():
                env.step(zero_action)
    except Exception as exc:
        print(f"[GT_DEMO] ERROR: {exc}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
