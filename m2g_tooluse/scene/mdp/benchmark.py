# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from isaaclab.utils.math import quat_apply_inverse, quat_rotate_inverse
import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import RigidObject
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import SceneEntityCfg

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import torch
from typing import Sequence
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


from isaaclab.managers import RecorderTerm
import torch

class InsertionSuccessRecorder(RecorderTerm):
    """Record insertion success rate (ISR)."""

    def record_post_step(self):
        # 调用你已经写好的 mdp 函数
        isr = mdp.insertion_success_rate(self.env)

        # 返回 (key, value)
        # key 支持层级，用 '/' 分割
        return "metrics/isr", isr


# -----------------------------------------------------------------------------
# (M1) Coordination Success Rate (CSR)
# -----------------------------------------------------------------------------

def coordination_success_rate(
    env: "ManagerBasedRLEnv",
    target_insert_num: int = 5,
    idle_action_threshold: float = 0.02,
    idle_steps: int = 10,
    agent_1: SceneEntityCfg = SceneEntityCfg("robot"),
    agent_2: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor | None:
    """
    Coordination Success Rate (CSR)

    Episode-level metric. Logged only when episode terminates.

    Success conditions:
    1. Inserted can count >= target_insert_num
    2. Exactly one agent enters idle state for >= idle_steps
    """

    # ---- only log at episode end ----
    if not env.termination_manager.is_terminated.any():
        return None

    device = env.device
    num_envs = env.num_envs

    robot_1: Articulation = env.scene[agent_1.name]
    robot_2: Articulation = env.scene[agent_2.name]

    action_1 = robot_1.data.applied_action
    action_2 = robot_2.data.applied_action

    mag_1 = torch.norm(action_1, dim=-1)
    mag_2 = torch.norm(action_2, dim=-1)

    # ---- initialize idle counters (once) ----
    if not hasattr(env, "_idle_counter"):
        env._idle_counter = {
            "robot_1": torch.zeros(num_envs, device=device),
            "robot_2": torch.zeros(num_envs, device=device),
        }

    env._idle_counter["robot_1"] = torch.where(
        mag_1 < idle_action_threshold,
        env._idle_counter["robot_1"] + 1,
        torch.zeros_like(env._idle_counter["robot_1"]),
    )

    env._idle_counter["robot_2"] = torch.where(
        mag_2 < idle_action_threshold,
        env._idle_counter["robot_2"] + 1,
        torch.zeros_like(env._idle_counter["robot_2"]),
    )

    idle_1 = env._idle_counter["robot_1"] >= idle_steps
    idle_2 = env._idle_counter["robot_2"] >= idle_steps

    coordination_ok = torch.logical_xor(idle_1, idle_2)

    # ---- reuse insertion logic for robustness ----
    inserted_ratio = insertion_success_rate(
        env,
        target_insert_num=target_insert_num,
    )
    target_reached = inserted_ratio >= 1.0

    csr = torch.logical_and(coordination_ok, target_reached)

    return csr.float()


# -----------------------------------------------------------------------------
# (M2) Insertion Success Rate (ISR)
# -----------------------------------------------------------------------------

def insertion_success_rate(
    env: "ManagerBasedRLEnv",
    target_insert_num: int = 5,
    can_cfgs: Sequence[SceneEntityCfg] | None = None,
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    container_center_xy: tuple[float, float] = (0.0, 0.0),
    container_half_extent_xy: tuple[float, float] = (0.16, 0.10),
    z_above_container: float = 0.01,
) -> torch.Tensor:
    """
    Insertion Success Rate (ISR)

    A can is considered inserted if:
    - XY inside container bounds
    - Z is above insertion reference plane

    NOTE:
    Z-direction assumption depends on container geometry.
    """

    if can_cfgs is None:
        can_cfgs = [SceneEntityCfg(f"can_{i}") for i in range(1, 9)]

    container: RigidObject = env.scene[container_cfg.name]
    cont_pos = container.data.root_pos_w
    cont_quat = container.data.root_quat_w

    num_envs = env.num_envs
    inserted = torch.zeros(num_envs, device=env.device)

    cx, cy = container_center_xy
    hx, hy = container_half_extent_xy

    for can_cfg in can_cfgs:
        can: RigidObject = env.scene[can_cfg.name]
        rel = can.data.root_pos_w - cont_pos
        rel_local = quat_apply_inverse(cont_quat, rel)

        in_xy = (
            (torch.abs(rel_local[:, 0] - cx) < hx)
            & (torch.abs(rel_local[:, 1] - cy) < hy)
        )
        in_z = rel_local[:, 2] >= z_above_container

        inserted += (in_xy & in_z).float()

    isr = torch.clamp(inserted / float(target_insert_num), 0.0, 1.0)
    return isr


# -----------------------------------------------------------------------------
# (M3) Grasp Success Rate (GSR)
# -----------------------------------------------------------------------------

def grasp_success_rate(
    env: "ManagerBasedRLEnv",
    can_cfgs: Sequence[SceneEntityCfg] | None = None,
    height_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Grasp Success Rate (GSR)

    Measures whether a can has EVER been lifted above height_threshold.
    """

    if can_cfgs is None:
        can_cfgs = [SceneEntityCfg(f"can_{i}") for i in range(1, 9)]

    num_envs = env.num_envs
    grasped = torch.zeros(num_envs, device=env.device)

    env_z = env.scene.env_origins[:, 2]

    for can_cfg in can_cfgs:
        can: RigidObject = env.scene[can_cfg.name]
        can_z = can.data.root_pos_w[:, 2] - env_z
        grasped += (can_z > height_threshold).float()

    gsr = grasped / float(len(can_cfgs))
    return gsr


# -----------------------------------------------------------------------------
# (M4) Place Success Rate (PSR)
# -----------------------------------------------------------------------------

def place_success_rate(
    env: "ManagerBasedRLEnv",
    can_cfgs: Sequence[SceneEntityCfg] | None = None,
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    container_center_xy: tuple[float, float] = (0.0, 0.0),
    container_half_extent_xy: tuple[float, float] = (0.16, 0.10),
    container_z_threshold: float = 0.10,
) -> torch.Tensor:
    """
    Place Success Rate (PSR)

    A can is considered placed if:
    - XY inside container bounds
    - Z below container top edge
    """

    if can_cfgs is None:
        can_cfgs = [SceneEntityCfg(f"can_{i}") for i in range(1, 9)]

    container: RigidObject = env.scene[container_cfg.name]
    cont_pos = container.data.root_pos_w
    cont_quat = container.data.root_quat_w

    num_envs = env.num_envs
    placed = torch.zeros(num_envs, device=env.device)

    cx, cy = container_center_xy
    hx, hy = container_half_extent_xy

    for can_cfg in can_cfgs:
        can: RigidObject = env.scene[can_cfg.name]
        rel = can.data.root_pos_w - cont_pos
        rel_local = quat_apply_inverse(cont_quat, rel)

        in_xy = (
            (torch.abs(rel_local[:, 0] - cx) < hx)
            & (torch.abs(rel_local[:, 1] - cy) < hy)
        )
        in_z = rel_local[:, 2] < container_z_threshold

        placed += (in_xy & in_z).float()

    psr = torch.clamp(placed / float(len(can_cfgs)), 0.0, 1.0)
    return psr
