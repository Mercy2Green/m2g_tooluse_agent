from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

from m2g_tooluse.assets.cfg.go2_piper import GO2PIPER_CFG


_DEFAULT_ASSET_ROOT = Path(os.environ.get("M2G_ASSET_ROOT", "assets"))

GO2PIPER_TRAIN_USD_PATH = str(
    _DEFAULT_ASSET_ROOT / "robots" / "Go2Piper" / "go2_piper_v1_train_sanitized.usd"
)

GO2PIPER_LIGHT_PIPER_USD_PATH = str(
    _DEFAULT_ASSET_ROOT / "robots" / "Go2Piper" / "go2_piper_v1_train_sanitized_light_piper.usd"
)


def _make_train_cfg(usd_path: str):
    cfg = deepcopy(GO2PIPER_CFG)
    cfg.spawn.usd_path = usd_path
    cfg.init_state.pos = (0.0, 0.0, 0.32)
    cfg.init_state.joint_pos = {
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
        "joint7": 0.0,
        "joint8": 0.0,
    }
    cfg.init_state.joint_vel = {".*": 0.0}
    return cfg


GO2PIPER_TRAIN_CFG = _make_train_cfg(GO2PIPER_TRAIN_USD_PATH)
GO2PIPER_LIGHT_PIPER_CFG = _make_train_cfg(GO2PIPER_LIGHT_PIPER_USD_PATH)
