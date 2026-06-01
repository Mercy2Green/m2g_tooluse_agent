# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef_to object,
    """

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )

def get_camera_rgba(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Fetch RGBA image from a camera sensor in the scene.
    Returns:
        (N, H, W, 4) torch.float32 in [0,1]
    """
    # get sensor name from cfg
    cam_name = sensor_cfg.name     # e.g., "main_camera"
    cam = env.scene.sensors[cam_name]

    # Ensure sensor is updated this frame
    # IsaacLab's sensors are auto-updated during env.step()

    # Fetch RGBA tensor
    # cam.data.output["rgb"] has shape [num_envs, H, W, 4]
    rgba = cam.data.output["rgb"]

    # # Convert to float32 [0,1]
    # if rgba.dtype != torch.float32:
    #     rgba = rgba.float() / 255.0

    return rgba

def get_cam_pos(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Fetch camera world-space position.

    Returns:
        (N, 3) torch.float32  -- camera position for each env
    """
    # get camera name
    cam_name = sensor_cfg.name
    cam = env.scene.sensors[cam_name]

    # Camera pose in world frame
    # cam.data.pos_w has shape (num_envs, 3)
    # print(cam.data)
    cam_pos_w = cam.data.pos_w

    # Subtract per-env origin (just like your get_cam_pos)
    cam_pos_env = cam_pos_w - env.scene.env_origins

    return cam_pos_env

def get_cam_quat(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Fetch camera world-space orientation as a quaternion.

    Returns:
        (N, 4) torch.float32  -- camera quaternion for each env
    """
    # get camera name
    cam_name = sensor_cfg.name
    cam = env.scene.sensors[cam_name]

    # Camera orientation in world frame
    # cam.data.quat_w has shape (num_envs, 4)
    cam_quat_w = cam.data.quat_w_world

    return cam_quat_w


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos

def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_robot_joint_state(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
) -> torch.Tensor:
    # hand_joint_names is a list of regex, use find_joints
    indexes, _ = env.scene["robot"].find_joints(joint_names)
    indexes = torch.tensor(indexes, dtype=torch.long)
    robot_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return robot_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos


# def get_finger_tip_positions(
#     env: ManagerBasedRLEnv,
#     link_names: list[str],
# ) -> torch.Tensor:
#     """
#     Return finger-tip positions for a list of link names.
#     Output: (num_envs, len(link_names)*3)
#     """
#     body_pos_w = env.scene["robot"].data.body_pos_w
#     origins = env.scene.env_origins

#     all_pos = []

#     for name in link_names:
#         idx = env.scene["robot"].data.body_names.index(name)
#         pos = body_pos_w[:, idx] - origins
#         all_pos.append(pos)

#     # concat along last dim: (N, 3*k)
#     return torch.cat(all_pos, dim=1)

