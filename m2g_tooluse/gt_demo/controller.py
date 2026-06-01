from __future__ import annotations

from typing import Any, Callable

from .search_and_pick import GtDemoSearchAndPick
from .types import SkillResult


class GtDemoController:
    """Facade used by script entry points and the ROS2 bridge.

    ROS2 callbacks should call this facade through the local RPC queue instead
    of embedding IsaacLab scene/action details directly in service handlers.
    """

    def __init__(self, env: Any, *, object_name: str = "object", enable_demo_attach_fallback: bool = True):
        self.env = env
        self.object_name = object_name
        self.demo = GtDemoSearchAndPick(env, enable_demo_attach_fallback=enable_demo_attach_fallback)
        self.last_result = SkillResult(success=True, message="gt_demo controller initialized")
        self.last_command = "idle"

    def start(self) -> SkillResult:
        self.last_command = "start"
        self.last_result = SkillResult(success=True, message="gt_demo is ready")
        return self.last_result

    def reset(self) -> SkillResult:
        self.last_command = "reset"
        self.env.reset()
        self.demo = GtDemoSearchAndPick(
            self.env,
            enable_demo_attach_fallback=self.demo.piper.enable_demo_attach_fallback,
        )
        self.last_result = SkillResult(success=True, message="gt_demo reset complete")
        return self.last_result

    def go_to_object(self) -> SkillResult:
        self.last_command = "go_to_object"
        found = self.demo.find_object(self.object_name)
        if not found.success:
            self.last_result = found
            return found
        self.last_result = self.demo.navigate_to_object(self.object_name)
        return self.last_result

    def pick_object(self) -> SkillResult:
        self.last_command = "pick_object"
        found = self.demo.find_object(self.object_name)
        if not found.success:
            self.last_result = found
            return found
        self.last_result = self.demo.pick_object(self.object_name)
        return self.last_result

    def run_full_demo(self) -> SkillResult:
        self.last_command = "run_full_demo"
        self.last_result = self.demo.search_and_pick(self.object_name)
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
            "go_to_object": self.go_to_object,
            "pick_object": self.pick_object,
            "run_full_demo": self.run_full_demo,
            "status": self.status,
        }
        handler = handlers.get(command)
        if handler is None:
            return SkillResult(success=False, message=f"unknown gt_demo command: {command}")
        return handler()
