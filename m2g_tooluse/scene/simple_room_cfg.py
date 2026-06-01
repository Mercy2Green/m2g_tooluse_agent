from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


_DEFAULT_ASSET_ROOT = Path(os.environ.get("M2G_ASSET_ROOT", "assets"))
ROOM_USD_PATH = str(_DEFAULT_ASSET_ROOT / "room" / "SimpleRoom_flatten.usd")


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
