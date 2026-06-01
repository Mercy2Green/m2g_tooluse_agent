from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def body_height_below_minimum(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any selected body/link drops below a minimum height.

    This works on articulated bodies and allows callers to specify the link/body
    via SceneEntityCfg.body_names instead of always using the root pose.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return torch.any(body_pos_w < minimum_height, dim=1)


def body_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the average selected body/link height from a target height."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    body_height = torch.mean(body_pos_w, dim=1)
    return torch.square(body_height - target_height)

def body_height_exp(
    env,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    body_z = torch.mean(body_z, dim=1)
    height_error = torch.square(body_z - target_height)
    return torch.exp(-height_error / std**2)