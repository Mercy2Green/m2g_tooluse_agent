#!/usr/bin/env python3
from __future__ import annotations

"""Go2 simple-room velocity teleop/debug script."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Go2 simple-room velocity teleop/debug.")
parser.add_argument("--task", type=str, default="M2G-ToolUse-Go2-SimpleRoom-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--mode", choices=("wasd", "scripted"), default="scripted")
parser.add_argument("--script", type=str, default="forward:0.2,turn_left:15,stop")
parser.add_argument("--forward", type=float, default=None)
parser.add_argument("--turn-left", type=float, default=None)
parser.add_argument(
    "--no-zero-action-step",
    action="store_true",
    default=False,
    help="Require an external locomotion action provider instead of debug zero-action stepping.",
)
parser.add_argument("--vx", type=float, default=0.25)
parser.add_argument("--vy", type=float, default=0.15)
parser.add_argument("--wz", type=float, default=0.4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import omni
import torch

import isaaclab_tasks  # noqa: F401
import m2g_tooluse.scene  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from m2g_tooluse.skills import Go2BaseNavSkill, IsaacLabVelocityCommandAdapter, SkillResult


class WasdKeyboard:
    """Small WASD velocity device using Omniverse keyboard events."""

    def __init__(self, vx: float, vy: float, wz: float):
        self.vx = vx
        self.vy = vy
        self.wz = wz
        self.command = [0.0, 0.0, 0.0]
        self.reset_requested = False
        self.quit_requested = False
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def close(self) -> None:
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def _on_event(self, event, *args, **kwargs) -> bool:
        sign = 0.0
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            sign = 1.0
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            sign = -1.0
        else:
            return True

        key = event.input.name
        if key == "W":
            self.command[0] += sign * self.vx
        elif key == "S":
            self.command[0] -= sign * self.vx
        elif key == "Q":
            self.command[1] += sign * self.vy
        elif key == "E":
            self.command[1] -= sign * self.vy
        elif key == "A":
            self.command[2] += sign * self.wz
        elif key == "D":
            self.command[2] -= sign * self.wz
        elif event.type == carb.input.KeyboardEventType.KEY_PRESS and key == "SPACE":
            self.command = [0.0, 0.0, 0.0]
        elif event.type == carb.input.KeyboardEventType.KEY_PRESS and key == "R":
            self.reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_PRESS and key == "ESCAPE":
            self.quit_requested = True
        return True


def _print_result(name: str, result: SkillResult) -> None:
    print(
        f"[SKILL] {name}: success={result.success} elapsed_s={result.elapsed_s:.3f} "
        f"message={result.message!r} command={result.command} target={result.target}",
        flush=True,
    )


def _script_items(args: argparse.Namespace) -> list[str]:
    if args.forward is not None or args.turn_left is not None:
        items = []
        if args.forward is not None:
            items.append(f"forward:{args.forward}")
        if args.turn_left is not None:
            items.append(f"turn_left:{args.turn_left}")
        items.append("stop")
        return items
    return [item.strip() for item in args.script.split(",") if item.strip()]


def run_scripted(nav: Go2BaseNavSkill, args: argparse.Namespace) -> None:
    handlers = {
        "forward": lambda value: nav.move_forward(float(value)),
        "backward": lambda value: nav.move_backward(float(value)),
        "strafe_left": lambda value: nav.strafe_left(float(value)),
        "strafe_right": lambda value: nav.strafe_right(float(value)),
        "turn_left": lambda value: nav.turn_left(float(value)),
        "turn_right": lambda value: nav.turn_right(float(value)),
        "turn_yaw": lambda value: nav.turn_yaw(float(value)),
        "stop": lambda value: nav.stop(),
    }
    for item in _script_items(args):
        name, _, value = item.partition(":")
        if name not in handlers:
            raise ValueError(f"unsupported scripted command: {name}")
        result = handlers[name](value)
        _print_result(name, result)
        if not result.success:
            raise RuntimeError(result.message)


def run_wasd(env, nav: Go2BaseNavSkill, args: argparse.Namespace) -> None:
    keyboard = WasdKeyboard(vx=args.vx, vy=args.vy, wz=args.wz)
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    try:
        while simulation_app.is_running() and not keyboard.quit_requested:
            with torch.inference_mode():
                vx, vy, wz = keyboard.command
                nav.walk_velocity(vx, vy, wz)
                if keyboard.reset_requested:
                    env.reset()
                    keyboard.reset_requested = False
                env.step(action)
    finally:
        keyboard.close()


def main() -> None:
    if args_cli.mode == "wasd" and args_cli.headless:
        raise RuntimeError("WASD mode requires a GUI; use --mode scripted when --headless is set")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Gym observation space: {env.observation_space}", flush=True)
    print(f"[INFO]: Gym action space: {env.action_space}", flush=True)
    env.reset()

    adapter = IsaacLabVelocityCommandAdapter(
        env,
        allow_zero_action_step=not args_cli.no_zero_action_step,
    )
    nav = Go2BaseNavSkill(adapter, limits=adapter.velocity_limits())

    try:
        if args_cli.mode == "scripted":
            run_scripted(nav, args_cli)
        else:
            run_wasd(env, nav, args_cli)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
