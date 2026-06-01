from __future__ import annotations

from typing import Any

from .skill.base_kinematic import KinematicDebugBaseController
from .skill.grasp_planner_gt import FixedObjectGraspPlanner
from .skill.object_pose_gt import IsaacGroundTruthObjectPoseProvider
from .skill.piper_pick_gt import PiperScriptedPickSkill
from .types import ObjectInfo, SkillResult


class GtDemoSearchAndPick:
    def __init__(self, env: Any, *, enable_demo_attach_fallback: bool = True):
        self.env = env
        self.pose_provider = IsaacGroundTruthObjectPoseProvider(env)
        self.base = KinematicDebugBaseController(env)
        self.grasp_planner = FixedObjectGraspPlanner()
        self.piper = PiperScriptedPickSkill(
            env,
            enable_demo_attach_fallback=enable_demo_attach_fallback,
        )
        self._last_object: ObjectInfo | None = None

    def find_object(self, name: str = "object") -> SkillResult:
        print("[GT_DEMO] find_object...")
        try:
            object_info = self.pose_provider.find_object(name)
            self._last_object = object_info
            return SkillResult(success=True, message=f"found {name}", data={"object_info": object_info})
        except Exception as exc:
            return SkillResult(success=False, message=f"find_object failed: {exc}")

    def navigate_to_object(self, name: str = "object") -> SkillResult:
        print("[GT_DEMO] navigate...")
        try:
            object_info = self._last_object or self.pose_provider.find_object(name)
            result = self.base.move_base_near_object(object_info)
            if not result.success:
                return result
            return SkillResult(success=True, message="navigate_to_grasp_standoff_gt complete", data=result.data)
        except Exception as exc:
            return SkillResult(success=False, message=f"navigate_to_object failed: {exc}")

    def pick_object(self, name: str = "object") -> SkillResult:
        print("[GT_DEMO] plan_grasp...")
        try:
            object_info = self._last_object or self.pose_provider.find_object(name)
            grasp_plan = self.grasp_planner.plan(object_info)
            print("[GT_DEMO] execute_pick...")
            result = self.piper.execute_pick(grasp_plan)
            result.data.setdefault("grasp_plan", grasp_plan)
            return result
        except Exception as exc:
            return SkillResult(success=False, message=f"pick_object failed: {exc}")

    def search_and_pick(self, name: str = "object") -> SkillResult:
        try:
            result = self.find_object(name)
            if not result.success:
                return result
            result = self.navigate_to_object(name)
            if not result.success:
                return result
            result = self.find_object(name)
            if not result.success:
                return result
            return self.pick_object(name)
        except Exception as exc:
            return SkillResult(success=False, message=f"search_and_pick failed: {exc}")
