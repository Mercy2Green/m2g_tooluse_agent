from __future__ import annotations

import os
from pathlib import Path

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from humanoid.tasks.data.can import CAN_FANTA_CFG
from humanoid.tasks.data.container import CONTAINER_2X3_CFG
from m2g_tooluse.assets.cfg.go2_piper import GO2PIPER_CFG

_DEFAULT_ASSET_ROOT = Path(os.environ.get("M2G_ASSET_ROOT", "assets"))
ROOM_USD_PATH = str(_DEFAULT_ASSET_ROOT / "room" / "SimpleRoom_flatten_low_desk.usd")


ROOM_CFG = AssetBaseCfg(
	prim_path="{ENV_REGEX_NS}/room",
	spawn=sim_utils.UsdFileCfg(
		usd_path=ROOM_USD_PATH,
		activate_contact_sensors=False,
		collision_props=sim_utils.CollisionPropertiesCfg(),
	),
	init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)

@configclass
class SimpleRoomSceneCfg(InteractiveSceneCfg):
	"""Simple room scene with room and insert-cans assets."""

	room: AssetBaseCfg = ROOM_CFG

@configclass
class GtDemoObjectTableSceneCfg(SimpleRoomSceneCfg):
    """Simple room scene for the isolated gt_demo.

    The target is currently a Fanta can asset, but the scene entity is named
    ``object`` so ROSA/tool code can remain object-agnostic for the first MVP.
    """

    object: RigidObjectCfg = CAN_FANTA_CFG.replace(
        prim_path="/World/envs/env_.*/Object",
    )
    object.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(-0.2, 0, -0.435),
        rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Keep the container as a visual/table-context landmark from the source scene.
    container: RigidObjectCfg = CONTAINER_2X3_CFG.replace(
        prim_path="/World/envs/env_.*/Container",
    )
    container.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(-0.2, 0.2, -0.38),
        rot=(1.0, 0.0, 0.0, 0.0),
    )

    robot: ArticulationCfg = GO2PIPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-2.0, 0.0, -0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                "joint7": 0.05,
                "joint8": -0.05,
            },
            joint_vel={".*": 0.0},
        ),
    )
