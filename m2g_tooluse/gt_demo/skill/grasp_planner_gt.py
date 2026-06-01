from __future__ import annotations

from m2g_tooluse.gt_demo.types import GraspPlan, ObjectInfo, Pose3D


class FixedObjectGraspPlanner:
    """Fixed can/object grasp planner for the first gt_demo MVP."""

    def plan(self, object_info: ObjectInfo) -> GraspPlan:
        ox, oy, oz = object_info.pose.position
        # Fixed quaternion in IsaacLab wxyz order. This must be adjusted once
        # the final Piper end-effector frame convention is selected.
        orientation = (1.0, 0.0, 0.0, 0.0)
        grasp = Pose3D(position=(ox - 0.08, oy, oz + 0.02), orientation=orientation)
        pregrasp = Pose3D(position=(ox - 0.18, oy, oz + 0.04), orientation=orientation)
        lift = Pose3D(position=(grasp.position[0], grasp.position[1], grasp.position[2] + 0.12), orientation=orientation)
        return GraspPlan(
            object_name=object_info.name,
            pregrasp_pose=pregrasp,
            grasp_pose=grasp,
            lift_pose=lift,
            gripper_open=0.05,
            gripper_closed=0.0,
            strategy="fixed_can_debug_grasp",
        )
