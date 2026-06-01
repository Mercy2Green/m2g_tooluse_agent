from __future__ import annotations

from typing import Any

from m2g_tooluse.gt_demo.types import ObjectInfo, Pose3D


def _to_float_tuple(values: Any) -> tuple[float, ...]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return tuple(float(value) for value in values)


class IsaacGroundTruthObjectPoseProvider:
    def __init__(self, env: Any, object_asset_name: str = "object", object_name: str = "object"):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.object_asset_name = object_asset_name
        self.object_name = object_name

    def find_object(self, name: str = "object") -> ObjectInfo:
        scene = getattr(self.unwrapped, "scene", None)
        if scene is None:
            raise RuntimeError("env has no IsaacLab scene")
        try:
            obj = scene[self.object_asset_name]
        except Exception as exc:
            raise RuntimeError(
                f"object asset '{self.object_asset_name}' not found in env.scene"
            ) from exc

        num_envs = int(getattr(self.unwrapped, "num_envs", 1))
        if num_envs > 1:
            print("[GT_DEMO] warning: num_envs > 1; using env_id=0 for ground-truth object pose")

        root_state = getattr(obj.data, "root_state_w", None)
        if root_state is None:
            raise RuntimeError(f"asset '{self.object_asset_name}' has no root_state_w data")

        pose_values = root_state[0, :7]
        position = _to_float_tuple(pose_values[:3])
        orientation = _to_float_tuple(pose_values[3:7])
        prim_path = getattr(getattr(obj, "cfg", None), "prim_path", "")
        object_name = name or self.object_name
        return ObjectInfo(
            name=object_name,
            prim_path=str(prim_path),
            pose=Pose3D(
                position=(position[0], position[1], position[2]),
                orientation=(orientation[0], orientation[1], orientation[2], orientation[3]),
                frame="world",
            ),
        )
