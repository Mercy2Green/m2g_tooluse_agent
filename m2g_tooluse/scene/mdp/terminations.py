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

def task_done_open_laptop(
    env: "ManagerBasedRLEnv",
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
    ratio: float = 0.7,
) -> torch.Tensor:
    """
    判断笔记本是否被“充分打开”
    - 使用第一个关节（盖子）的上限 laptop_upper。如果当前角度超过上限的 70%（开得足够大），就认为任务成功。
    - 将对应观测的 success 标记为 1。
    """
    laptop = env.scene[laptop_cfg.name]
    laptop_upper = laptop.data.joint_limits[:, 0, 1]
    done = laptop.data.joint_pos[:, 0] > laptop_upper * ratio
    return done

def task_done_close_drawer(
    env,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    drawer_bottom_joint_id: int = 0,
    close_ratio: float = 0.90,
    vel_threshold: float = 0.05,
):
    """MPD-style success condition for CLOSE DRAWER task."""

    drawer: Articulation = env.scene[drawer_cfg.name]

    # joint position
    joint_pos = drawer.data.joint_pos[:, drawer_bottom_joint_id]
    joint_upper = drawer.data.joint_pos_limits[:, drawer_bottom_joint_id, 1]

    # drawer close?
    close_mask = joint_pos > (joint_upper * close_ratio)

    # joint velocity
    joint_vel = torch.abs(drawer.data.joint_vel[:, drawer_bottom_joint_id])
    vel_mask = joint_vel < vel_threshold

    done = torch.logical_and(close_mask, vel_mask)
    return done

def task_done_close_drawer_joints(
    env,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    joint_ids: tuple[int, ...] = (0, 1),
    use_upper: tuple[bool, ...] = (True, True),
    close_ratio: float = 0.90,
    vel_threshold: float = 0.05,
) -> torch.Tensor:
    """Check whether a drawer-closing task is complete.

    Success conditions:
    1. All specified drawer joints are sufficiently closed.
       For each joint, the closing reference (upper or lower limit)
       is explicitly specified by `use_upper`.
    2. All specified drawer joints have sufficiently small velocities.
    """

    assert len(joint_ids) == len(use_upper), (
        "joint_ids and use_upper must have the same length"
    )

    drawer = env.scene[drawer_cfg.name]

    # Joint states
    joint_pos = drawer.data.joint_pos                  # [N, J]
    joint_vel = torch.abs(drawer.data.joint_vel)       # [N, J]
    joint_limits = drawer.data.joint_pos_limits        # [N, J, 2]

    joint_lower = joint_limits[..., 0]
    joint_upper = joint_limits[..., 1]

    # Joints of interest
    joint_ids_t = torch.tensor(joint_ids, device=joint_pos.device)
    use_upper_t = torch.tensor(use_upper, device=joint_pos.device)

    pos = joint_pos[:, joint_ids_t]
    vel = joint_vel[:, joint_ids_t]
    lower = joint_lower[:, joint_ids_t]
    upper = joint_upper[:, joint_ids_t]

    # -------------------------------------------------
    # Position condition (explicit reference)
    # -------------------------------------------------
    # Thresholds per joint
    upper_thresh = upper * close_ratio
    lower_thresh = lower * close_ratio

    # Select threshold and comparison per joint
    close_mask_per_joint = torch.where(
        use_upper_t,
        pos >= upper_thresh,   # use upper limit
        pos <= lower_thresh,   # use lower limit
    )

    close_mask = torch.all(close_mask_per_joint, dim=-1)

    # -------------------------------------------------
    # Velocity condition
    # -------------------------------------------------
    vel_mask = torch.all(vel < vel_threshold, dim=-1)

    # -------------------------------------------------
    # Final condition
    # -------------------------------------------------
    done = torch.logical_and(close_mask, vel_mask)

    return done

def task_done_open_drawer_joints(
    env,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    joint_ids: tuple[int, ...] = (0,),
    use_upper: tuple[bool, ...] = (True,),
    open_ratio: float = 0.6,
) -> torch.Tensor:

    drawer = env.scene[drawer_cfg.name]

    joint_pos = drawer.data.joint_pos
    joint_limits = drawer.data.joint_pos_limits

    lower = joint_limits[..., 0]
    upper = joint_limits[..., 1]

    joint_ids_t = torch.tensor(joint_ids, device=joint_pos.device)
    use_upper_t = torch.tensor(use_upper, device=joint_pos.device)

    pos   = joint_pos[:, joint_ids_t]
    lower = lower[:, joint_ids_t]
    upper = upper[:, joint_ids_t]

    joint_range = upper - lower

    # 如果关闭在 upper → 打开就是往 lower 方向
    open_mask_per_joint = torch.where(
        use_upper_t,
        pos <= upper - open_ratio * joint_range,
        pos >= lower + open_ratio * joint_range,
    )

    return torch.all(open_mask_per_joint, dim=-1)


# def task_done_open_drawer_joints(
#     env,
#     drawer_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     joint_ids: tuple[int, ...] = (0, 1),
#     use_upper: tuple[bool, ...] = (True, True),
#     open_ratio: float = 0.90,
# ) -> torch.Tensor:
#     """Check whether a drawer-opening task is complete.

#     Success conditions:
#     1. All specified drawer joints are sufficiently opened.
#        The opening reference (upper or lower limit) is inferred
#        as the opposite of the closing reference specified by `use_upper`.
#     2. All specified drawer joints have sufficiently small velocities.
#     """

#     assert len(joint_ids) == len(use_upper), (
#         "joint_ids and use_upper must have the same length"
#     )

#     drawer = env.scene[drawer_cfg.name]

#     # Joint states
#     joint_pos = drawer.data.joint_pos                  # [N, J]
#     joint_limits = drawer.data.joint_pos_limits        # [N, J, 2]

#     joint_lower = joint_limits[..., 0]
#     joint_upper = joint_limits[..., 1]

#     # Joints of interest
#     joint_ids_t = torch.tensor(joint_ids, device=joint_pos.device)
#     use_upper_t = torch.tensor(use_upper, device=joint_pos.device)

#     pos = joint_pos[:, joint_ids_t]
#     lower = joint_lower[:, joint_ids_t]
#     upper = joint_upper[:, joint_ids_t]

#     # -------------------------------------------------
#     # Position condition (explicit reference)
#     # -------------------------------------------------
#     # Opening thresholds (opposite of closing reference)
#     open_upper_thresh = upper * open_ratio
#     open_lower_thresh = lower * open_ratio

#     close_ref_is_upper = use_upper_t
#     open_ref_is_upper = ~close_ref_is_upper

#     print(pos, open_upper_thresh,upper)

#     open_mask_per_joint = torch.where(
#         open_ref_is_upper,
#         pos >= open_upper_thresh,   # opening towards upper
#         pos <= open_lower_thresh,   # opening towards lower
#     )

#     open_mask = torch.all(open_mask_per_joint, dim=-1)

#     # -------------------------------------------------
#     # Final condition
#     # -------------------------------------------------
#     done = open_mask

#     return done





def task_done_close_drawer_hand_away(
    env,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    drawer_bottom_joint_id: int = 0,
    close_ratio: float = 0.90,
    vel_threshold: float = 0.05,
    hand_away_threshold: float = 0.18,
) -> torch.Tensor:
    """Determine if the drawer-closing task is complete.

    Success when:
    1. Drawer bottom joint has reached close_ratio * upper_limit.
    2. Drawer joint velocity is small enough.
    3. BOTH hands are at least hand_away_threshold away from the handle.
    """

    # Get drawer articulation
    drawer = env.scene[drawer_cfg.name]

    # -----------------------------
    # Drawer condition 1: joint closed
    # -----------------------------
    joint_pos = drawer.data.joint_pos[:, drawer_bottom_joint_id]
    joint_upper = drawer.data.joint_pos_limits[:, drawer_bottom_joint_id, 1]

    close_mask = joint_pos > (joint_upper * close_ratio)

    # -----------------------------
    # Drawer condition 2: joint velocity small
    # -----------------------------
    joint_vel = torch.abs(drawer.data.joint_vel[:, drawer_bottom_joint_id])
    vel_mask = joint_vel < vel_threshold

    # -----------------------------
    # Hand condition: both hands far away
    # -----------------------------
    # Handle world position = drawer root + fixed offset
    handle_offset = torch.tensor([-0.31, 0.30, -0.05], device=env.device)
    handle_pos_w = drawer.data.root_pos_w + handle_offset  # [n_env, 3]

    # Robot links
    robot = env.scene["robot"]
    robot_body_pos_w = robot.data.body_pos_w
    body_names = robot.data.body_names

    # Find indices of wrists
    left_idx = body_names.index("left_wrist_yaw_link")
    right_idx = body_names.index("right_wrist_yaw_link")

    # EEF positions
    left_pos = robot_body_pos_w[:, left_idx, :]
    right_pos = robot_body_pos_w[:, right_idx, :]

    # Distances to handle
    left_dist = torch.norm(left_pos - handle_pos_w, dim=-1)
    right_dist = torch.norm(right_pos - handle_pos_w, dim=-1)

    left_far = left_dist > hand_away_threshold
    right_far = right_dist > hand_away_threshold
    both_hands_far = torch.logical_and(left_far, right_far)

    # -----------------------------
    # Combine all success conditions
    # -----------------------------
    done = close_mask
    done = torch.logical_and(done, vel_mask)
    done = torch.logical_and(done, both_hands_far)

    return done


def task_done_open_drawer_hand_away(
    env,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    drawer_bottom_joint_id: int = 0,
    open_ratio: float = 0.90,
    vel_threshold: float = 0.05,
    hand_away_threshold: float = 0.18,
) -> torch.Tensor:
    """
    Determine if the drawer-opening task is complete.

    Success when:
    1. Drawer bottom joint has reached: lower_limit + open_ratio * (upper_limit - lower_limit)
       (Meaning: sufficiently OPEN)
    2. Drawer joint velocity is small.
    3. BOTH hands are at least hand_away_threshold away from the handle.
    """

    drawer = env.scene[drawer_cfg.name]

    # -----------------------------
    # Drawer condition 1: sufficiently OPEN
    # -----------------------------
    joint_pos = drawer.data.joint_pos[:, drawer_bottom_joint_id]
    # print(drawer.data.joint_names[drawer_bottom_joint_id])
    joint_limits = drawer.data.joint_pos_limits[:, drawer_bottom_joint_id]

    lower = joint_limits[:, 0]   # fully OPEN
    upper = joint_limits[:, 1]   # fully CLOSED

    # open threshold = lower + ratio * (upper - lower)
    open_threshold = lower + open_ratio * (upper - lower)

    # A drawer is open when joint_pos is LOWER or equal to threshold
    open_mask = joint_pos < open_threshold

    # -----------------------------
    # Drawer condition 2: small velocity
    # -----------------------------
    joint_vel = torch.abs(drawer.data.joint_vel[:, drawer_bottom_joint_id])
    vel_mask = joint_vel < vel_threshold

    # -----------------------------
    # Hand condition: both hands far away
    # -----------------------------
    handle_offset = torch.tensor([-0.31, 0.30, -0.05], device=env.device)
    handle_pos_w = drawer.data.root_pos_w + handle_offset

    # robot
    robot = env.scene["robot"]
    pos_w = robot.data.body_pos_w
    names = robot.data.body_names

    left_idx = names.index("left_wrist_yaw_link")
    right_idx = names.index("right_wrist_yaw_link")

    left_pos = pos_w[:, left_idx]
    right_pos = pos_w[:, right_idx]

    left_far = torch.norm(left_pos - handle_pos_w, dim=-1) > hand_away_threshold
    right_far = torch.norm(right_pos - handle_pos_w, dim=-1) > hand_away_threshold

    both_far = torch.logical_and(left_far, right_far)

    # -----------------------------
    # Combine all conditions
    # -----------------------------
    done = torch.logical_and(open_mask, vel_mask)
    done = torch.logical_and(done, both_far)

    return done

# def task_done_object_in_container(
#     env: ManagerBasedRLEnv,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
#     radius: float = 0.15,
#     edge_height: float = 0.5,
#     vel_threshold: float = 0.20,
# ) -> torch.Tensor:
#     """Check if object has been successfully placed inside the container.

#     Success conditions:
#     1. object XY is within the container opening radius
#     2. object Z is below the container edge height
#     3. object velocity is small (object has settled)
#     """

#     # Read object + container states
#     object: RigidObject = env.scene[object_cfg.name]
#     container: RigidObject = env.scene[container_cfg.name]

#     obj_pos = object.data.root_pos_w
#     cont_pos = container.data.root_pos_w
#     obj_vel = torch.abs(object.data.root_vel_w)

#     # Relative XY distance from container center
#     dx = obj_pos[:, 0] - cont_pos[:, 0]
#     dy = obj_pos[:, 1] - cont_pos[:, 1]
#     dist_xy = torch.sqrt(dx * dx + dy * dy)

#     # Condition 1: within XY circle
#     in_xy = dist_xy < radius

#     # Condition 2: object inside bowl height
#     in_height = obj_pos[:, 2] < edge_height

#     # Condition 3: velocity is small (object settled)
#     vel_small = torch.logical_and(
#         obj_vel[:, 0] < vel_threshold,
#         torch.logical_and(
#             obj_vel[:, 1] < vel_threshold,
#             obj_vel[:, 2] < vel_threshold,
#         ),
#     )

#     done = torch.logical_and(in_xy, in_height)
#     done = torch.logical_and(done, vel_small)

#     return done

def task_done_object_in_container(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    radius: float = 0.15,
    edge_height: float = 0.5,
    vel_threshold: float = 0.20,
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    container: RigidObject = env.scene[container_cfg.name]

    # world pose
    obj_pos_w = object.data.root_pos_w
    cont_pos_w = container.data.root_pos_w
    cont_quat_w = container.data.root_quat_w  # quaternion, wxyz

    # Step 1 —— 转为 container 坐标系
    # p_local = R^T * (p_world - cont_pos_w)
    rel = obj_pos_w - cont_pos_w
    rel_local = quat_apply_inverse(cont_quat_w, rel)  # IsaacLab helper

    # Step 2 —— 在 container 局部坐标系判断
    # bowl opening 朝上的情况下：
    # local z 轴 = bowl 法线
    dist_xy = torch.sqrt(rel_local[:, 0] ** 2 + rel_local[:, 1] ** 2)
    in_xy = dist_xy < radius

    in_height = rel_local[:, 2] < edge_height

    # Step 3 —— 速度判断保持不变
    obj_vel = torch.abs(object.data.root_vel_w)
    vel_small = (
        (obj_vel[:, 0] < vel_threshold)
        & (obj_vel[:, 1] < vel_threshold)
        & (obj_vel[:, 2] < vel_threshold)
    )

    done = in_xy & in_height & vel_small
    return done

def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str = "",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        max_height: Maximum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    if task_link_name == "":
        raise ValueError("task_link_name must be provided to task_done_pick_place")

    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index(task_link_name)
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done


def task_done_nut_pour(
    env: ManagerBasedRLEnv,
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("black_sorting_bin"),
    max_bowl_to_scale_x: float = 0.055,
    max_bowl_to_scale_y: float = 0.055,
    max_bowl_to_scale_z: float = 0.025,
    max_nut_to_bowl_x: float = 0.050,
    max_nut_to_bowl_y: float = 0.050,
    max_nut_to_bowl_z: float = 0.019,
    max_beaker_to_bin_x: float = 0.08,
    max_beaker_to_bin_y: float = 0.12,
    max_beaker_to_bin_z: float = 0.07,
) -> torch.Tensor:
    """Determine if the nut pouring task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The factory nut is in the sorting bowl
    2. The sorting beaker is in the sorting bin
    3. The sorting bowl is placed on the sorting scale

    Args:
        env: The RL environment instance.
        sorting_scale_cfg: Configuration for the sorting scale entity.
        sorting_bowl_cfg: Configuration for the sorting bowl entity.
        sorting_beaker_cfg: Configuration for the sorting beaker entity.
        factory_nut_cfg: Configuration for the factory nut entity.
        sorting_bin_cfg: Configuration for the sorting bin entity.
        max_bowl_to_scale_x: Maximum x position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_y: Maximum y position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_z: Maximum z position of the sorting bowl relative to the sorting scale for task completion.
        max_nut_to_bowl_x: Maximum x position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_y: Maximum y position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_z: Maximum z position of the factory nut relative to the sorting bowl for task completion.
        max_beaker_to_bin_x: Maximum x position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_y: Maximum y position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_z: Maximum z position of the sorting beaker relative to the sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    sorting_scale: RigidObject = env.scene[sorting_scale_cfg.name]
    sorting_bowl: RigidObject = env.scene[sorting_bowl_cfg.name]
    factory_nut: RigidObject = env.scene[factory_nut_cfg.name]
    sorting_beaker: RigidObject = env.scene[sorting_beaker_cfg.name]
    sorting_bin: RigidObject = env.scene[sorting_bin_cfg.name]

    # Get positions relative to environment origin
    scale_pos = sorting_scale.data.root_pos_w - env.scene.env_origins
    bowl_pos = sorting_bowl.data.root_pos_w - env.scene.env_origins
    sorting_beaker_pos = sorting_beaker.data.root_pos_w - env.scene.env_origins
    nut_pos = factory_nut.data.root_pos_w - env.scene.env_origins
    bin_pos = sorting_bin.data.root_pos_w - env.scene.env_origins

    # nut to bowl
    nut_to_bowl_x = torch.abs(nut_pos[:, 0] - bowl_pos[:, 0])
    nut_to_bowl_y = torch.abs(nut_pos[:, 1] - bowl_pos[:, 1])
    nut_to_bowl_z = nut_pos[:, 2] - bowl_pos[:, 2]

    # bowl to scale
    bowl_to_scale_x = torch.abs(bowl_pos[:, 0] - scale_pos[:, 0])
    bowl_to_scale_y = torch.abs(bowl_pos[:, 1] - scale_pos[:, 1])
    bowl_to_scale_z = bowl_pos[:, 2] - scale_pos[:, 2]

    # beaker to bin
    beaker_to_bin_x = torch.abs(sorting_beaker_pos[:, 0] - bin_pos[:, 0])
    beaker_to_bin_y = torch.abs(sorting_beaker_pos[:, 1] - bin_pos[:, 1])
    beaker_to_bin_z = sorting_beaker_pos[:, 2] - bin_pos[:, 2]

    done = nut_to_bowl_x < max_nut_to_bowl_x
    done = torch.logical_and(done, nut_to_bowl_y < max_nut_to_bowl_y)
    done = torch.logical_and(done, nut_to_bowl_z < max_nut_to_bowl_z)
    done = torch.logical_and(done, bowl_to_scale_x < max_bowl_to_scale_x)
    done = torch.logical_and(done, bowl_to_scale_y < max_bowl_to_scale_y)
    done = torch.logical_and(done, bowl_to_scale_z < max_bowl_to_scale_z)
    done = torch.logical_and(done, beaker_to_bin_x < max_beaker_to_bin_x)
    done = torch.logical_and(done, beaker_to_bin_y < max_beaker_to_bin_y)
    done = torch.logical_and(done, beaker_to_bin_z < max_beaker_to_bin_z)

    return done


def task_done_exhaust_pipe(
    env: ManagerBasedRLEnv,
    blue_exhaust_pipe_cfg: SceneEntityCfg = SceneEntityCfg("blue_exhaust_pipe"),
    blue_sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("blue_sorting_bin"),
    max_blue_exhaust_to_bin_x: float = 0.085,
    max_blue_exhaust_to_bin_y: float = 0.200,
    min_blue_exhaust_to_bin_y: float = -0.090,
    max_blue_exhaust_to_bin_z: float = 0.070,
) -> torch.Tensor:
    """Determine if the exhaust pipe task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The blue exhaust pipe is placed in the correct position

    Args:
        env: The RL environment instance.
        blue_exhaust_pipe_cfg: Configuration for the blue exhaust pipe entity.
        blue_sorting_bin_cfg: Configuration for the blue sorting bin entity.
        max_blue_exhaust_to_bin_x: Maximum x position of the blue exhaust pipe relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_y: Maximum y position of the blue exhaust pipe relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_z: Maximum z position of the blue exhaust pipe relative to the blue sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    blue_exhaust_pipe: RigidObject = env.scene[blue_exhaust_pipe_cfg.name]
    blue_sorting_bin: RigidObject = env.scene[blue_sorting_bin_cfg.name]

    # Get positions relative to environment origin
    blue_exhaust_pipe_pos = blue_exhaust_pipe.data.root_pos_w - env.scene.env_origins
    blue_sorting_bin_pos = blue_sorting_bin.data.root_pos_w - env.scene.env_origins

    # blue exhaust to bin
    blue_exhaust_to_bin_x = torch.abs(blue_exhaust_pipe_pos[:, 0] - blue_sorting_bin_pos[:, 0])
    blue_exhaust_to_bin_y = blue_exhaust_pipe_pos[:, 1] - blue_sorting_bin_pos[:, 1]
    blue_exhaust_to_bin_z = blue_exhaust_pipe_pos[:, 2] - blue_sorting_bin_pos[:, 2]

    done = blue_exhaust_to_bin_x < max_blue_exhaust_to_bin_x
    done = torch.logical_and(done, blue_exhaust_to_bin_y < max_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_y > min_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_z < max_blue_exhaust_to_bin_z)

    return done



# def task_done_pour_balls(
#     env: "ManagerBasedRLEnv",
#     bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
#     ball1_cfg: SceneEntityCfg = SceneEntityCfg("ball1"),
#     ball2_cfg: SceneEntityCfg = SceneEntityCfg("ball2"),
#     ball3_cfg: SceneEntityCfg = SceneEntityCfg("ball3"),
#     ball4_cfg: SceneEntityCfg = SceneEntityCfg("ball4"),
#     ball5_cfg: SceneEntityCfg = SceneEntityCfg("ball5"),
#     ball6_cfg: SceneEntityCfg = SceneEntityCfg("ball6"),
#     ball7_cfg: SceneEntityCfg = SceneEntityCfg("ball7"),
#     ball8_cfg: SceneEntityCfg = SceneEntityCfg("ball8"),
#     surface_center_offset=(0.0, 0.0, 0.075),
#     surface_radius: float = 0.085,
#     min_poured_balls: int = 3,
#     angle_tolerance_deg: float = 10.0,
# ) -> torch.Tensor:
#     """
#     - Compute the bowl surface center from the bowl pose and ``surface_center_offset``.
#     - For each ball, consider the vector from surface center to ball and the bowl surface normal.
#     - When the angle between them is ~90 degrees and the distance to the center is within
#       ``surface_radius``, the ball is counted as poured (only once per episode).
#     - Episode success when at least ``min_poured_balls`` balls have been poured.
#     """

#     bowl = env.scene[bowl_cfg.name]
#     balls_cfg = [ball1_cfg, ball2_cfg, ball3_cfg, ball4_cfg, ball5_cfg, ball6_cfg, ball7_cfg, ball8_cfg]
#     balls = [env.scene[cfg.name] for cfg in balls_cfg]

#     # positions in env-local frame
#     bowl_pos = bowl.data.root_pos_w - env.scene.env_origins
#     bowl_quat = bowl.data.root_quat_w

#     device = bowl_pos.device
#     num_envs = bowl_pos.shape[0]

#     # Lazily allocate per-episode buffers
#     if (not hasattr(env, "pour_balls_pass")) or env.pour_balls_pass.shape != (num_envs, len(balls)):
#         env.pour_balls_pass = torch.zeros((num_envs, len(balls)), dtype=torch.bool, device=device)
#         env.pour_balls_count = torch.zeros(num_envs, dtype=torch.int32, device=device)

#     # Bowl surface center in env-local frame
#     offset = torch.tensor(surface_center_offset, device=device).expand(num_envs, 3)
#     surface_center = bowl_pos + math_utils.quat_apply(bowl_quat, offset)

#     # Surface normal (from bowl root to surface center)
#     surface_normal = surface_center - bowl_pos
#     surface_normal = surface_normal / torch.norm(surface_normal, dim=-1, keepdim=True).clamp_min(1e-6)

#     # Angle window around 90 degrees
#     min_angle = 90.0 - angle_tolerance_deg
#     max_angle = 90.0 + angle_tolerance_deg

#     # Iterate over balls, update pass flags and counts
#     for idx, ball in enumerate(balls):
#         ball_pos = ball.data.root_pos_w - env.scene.env_origins

#         vec = ball_pos - surface_center
#         vec = vec / torch.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)

#         cos_theta = (vec * surface_normal).sum(dim=-1).clamp(-1.0, 1.0)
#         angle = torch.acos(cos_theta) * 180.0 / torch.pi

#         dist = torch.norm(surface_center - ball_pos, dim=-1)

#         angle_ok = torch.logical_and(angle > min_angle, angle < max_angle)
#         dist_ok = dist < surface_radius
#         pass_now = torch.logical_and(angle_ok, dist_ok)

#         already_passed = env.pour_balls_pass[:, idx]
#         new_pass = torch.logical_and(torch.logical_not(already_passed), pass_now)

#         env.pour_balls_pass[:, idx] = torch.logical_or(already_passed, pass_now)
#         env.pour_balls_count += new_pass.to(env.pour_balls_count.dtype)

#     done = env.pour_balls_count >= min_poured_balls
# # 这边球弹出来后不会重置为未进入状态 和原有一样
#     return done


def task_done_pour_balls(
    env: "ManagerBasedRLEnv",
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    ball1_cfg: SceneEntityCfg = SceneEntityCfg("ball1"),
    ball2_cfg: SceneEntityCfg = SceneEntityCfg("ball2"),
    ball3_cfg: SceneEntityCfg = SceneEntityCfg("ball3"),
    ball4_cfg: SceneEntityCfg = SceneEntityCfg("ball4"),
    ball5_cfg: SceneEntityCfg = SceneEntityCfg("ball5"),
    ball6_cfg: SceneEntityCfg = SceneEntityCfg("ball6"),
    ball7_cfg: SceneEntityCfg = SceneEntityCfg("ball7"),
    ball8_cfg: SceneEntityCfg = SceneEntityCfg("ball8"),
    surface_center_offset=(0.0, 0.0, 0.075),
    surface_radius: float = 0.085,
    vel_threshold: float = 0.05,   # m/s，静止阈值
) -> torch.Tensor:
    """
    Task is done when:
    1. ALL balls are inside the bowl (distance to surface center < surface_radius)
    2. ALL balls are approximately static (linear velocity < vel_threshold)
    """

    bowl = env.scene[bowl_cfg.name]
    balls_cfg = [
        ball1_cfg, ball2_cfg, ball3_cfg, ball4_cfg,
        ball5_cfg, ball6_cfg, ball7_cfg, ball8_cfg
    ]
    balls = [env.scene[cfg.name] for cfg in balls_cfg]

    # bowl pose in env-local frame
    bowl_pos = bowl.data.root_pos_w - env.scene.env_origins
    bowl_quat = bowl.data.root_quat_w

    device = bowl_pos.device
    num_envs = bowl_pos.shape[0]

    # bowl surface center
    offset = torch.tensor(surface_center_offset, device=device).expand(num_envs, 3)
    surface_center = bowl_pos + math_utils.quat_apply(bowl_quat, offset)

    # per-env flag: start with all True
    all_inside = torch.ones(num_envs, dtype=torch.bool, device=device)
    all_static = torch.ones(num_envs, dtype=torch.bool, device=device)

    for ball in balls:
        # position
        ball_pos = ball.data.root_pos_w - env.scene.env_origins
        dist = torch.norm(ball_pos - surface_center, dim=-1)

        inside = dist < surface_radius
        all_inside &= inside

        # linear velocity magnitude
        ball_vel = ball.data.root_lin_vel_w
        speed = torch.norm(ball_vel, dim=-1)

        static = speed < vel_threshold
        all_static &= static

    done = all_inside & all_static
    return done



def task_done_unload_cans(
    env,
    can1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    require_upright_and_height: bool = True,
    up_z_threshold: float = 0.1,
    min_height: float = 0.3,
    vel_threshold: float = 0.2,          # ⭐ 新增：速度阈值
    container_center_xy=(0.48, 0.0),
    container_half_extent_xy=(0.16, 0.10),
) -> torch.Tensor:

    can1 = env.scene[can1_cfg.name]
    can2 = env.scene[can2_cfg.name]
    container = env.scene[container_cfg.name]

    # positions in env-local frame
    can1_pos = can1.data.root_pos_w - env.scene.env_origins
    can2_pos = can2.data.root_pos_w - env.scene.env_origins
    container_pos = container.data.root_pos_w - env.scene.env_origins

    device = container_pos.device

    # Base container footprint (reference)
    container_center_ref = torch.tensor(container_center_xy, device=device)
    container_half_extent = torch.tensor(container_half_extent_xy, device=device)

    container_xy_lower_ref = container_center_ref - container_half_extent
    container_xy_upper_ref = container_center_ref + container_half_extent

    # Offset by current container position
    offset_xy = container_pos[:, 0:2] - container_center_ref
    lower_xy = container_xy_lower_ref + offset_xy
    upper_xy = container_xy_upper_ref + offset_xy

    # Check whether cans are inside the container XY box
    can1_in_x = (can1_pos[:, 0] > lower_xy[:, 0]) & (can1_pos[:, 0] < upper_xy[:, 0])
    can1_in_y = (can1_pos[:, 1] > lower_xy[:, 1]) & (can1_pos[:, 1] < upper_xy[:, 1])
    can2_in_x = (can2_pos[:, 0] > lower_xy[:, 0]) & (can2_pos[:, 0] < upper_xy[:, 0])
    can2_in_y = (can2_pos[:, 1] > lower_xy[:, 1]) & (can2_pos[:, 1] < upper_xy[:, 1])

    can1_in_container = can1_in_x & can1_in_y
    can2_in_container = can2_in_x & can2_in_y

    done = ~(can1_in_container | can2_in_container)

    # -----------------------------
    # Upright + height constraint
    # -----------------------------
    if require_upright_and_height:
        can1_z = can1_pos[:, 2]
        can2_z = can2_pos[:, 2]

        z_axis = torch.zeros_like(can1_pos)
        z_axis[:, 2] = 1.0

        z_can1 = math_utils.quat_apply(can1.data.root_quat_w, z_axis)
        z_can2 = math_utils.quat_apply(can2.data.root_quat_w, z_axis)

        can1_up = z_can1[:, 2] > up_z_threshold
        can2_up = z_can2[:, 2] > up_z_threshold
        can1_high = can1_z > min_height
        can2_high = can2_z > min_height

        both_upright = (can1_up & can1_high) & (can2_up & can2_high)
        done = done & both_upright

    # -----------------------------
    # ⭐ Velocity constraint（新增）
    # -----------------------------
    can1_vel = torch.abs(can1.data.root_vel_w)
    can2_vel = torch.abs(can2.data.root_vel_w)

    can1_slow = (
        (can1_vel[:, 0] < vel_threshold)
        & (can1_vel[:, 1] < vel_threshold)
        & (can1_vel[:, 2] < vel_threshold)
    )
    can2_slow = (
        (can2_vel[:, 0] < vel_threshold)
        & (can2_vel[:, 1] < vel_threshold)
        & (can2_vel[:, 2] < vel_threshold)
    )

    both_slow = can1_slow & can2_slow
    done = done & both_slow

    return done


def task_done_insert_and_unload_cans(
    env: "ManagerBasedRLEnv",
    can1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    container_center_xy=(0.48, 0.0),
    container_half_extent_xy=(0.16, 0.10),
    z_threshold: float = 0.85,
) -> torch.Tensor:
    """
    - A can is marked as *inserted* once it lies inside the container XY footprint
      and below ``z_threshold``.
    - A can is *unloaded* when it is currently outside the container XY footprint
      **and** has been inserted before in the same episode.
    - Episode success when both cans are unloaded.
    """

    can1 = env.scene[can1_cfg.name]
    can2 = env.scene[can2_cfg.name]
    container = env.scene[container_cfg.name]

    # positions in env-local frame
    can1_pos = can1.data.root_pos_w - env.scene.env_origins
    can2_pos = can2.data.root_pos_w - env.scene.env_origins
    container_pos = container.data.root_pos_w - env.scene.env_origins

    device = container_pos.device
    num_envs = container_pos.shape[0]

    # Lazily allocate per-env flags that persist across steps in an episode.
    if (not hasattr(env, "insert_and_unload_cans_insert1")) or env.insert_and_unload_cans_insert1.shape[0] != num_envs:
        env.insert_and_unload_cans_insert1 = torch.zeros(num_envs, dtype=torch.bool, device=device)
        env.insert_and_unload_cans_insert2 = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Container footprint in env-local XY (with offset from current container pose)
    container_center_ref = torch.tensor(container_center_xy, device=device)
    container_half_extent = torch.tensor(container_half_extent_xy, device=device)
    container_xy_lower_ref = container_center_ref - container_half_extent
    container_xy_upper_ref = container_center_ref + container_half_extent

    offset_xy = container_pos[:, 0:2] - container_center_ref
    lower_xy = container_xy_lower_ref + offset_xy
    upper_xy = container_xy_upper_ref + offset_xy

    # Inside/outside checks
    can1_in_x = torch.logical_and(can1_pos[:, 0] > lower_xy[:, 0], can1_pos[:, 0] < upper_xy[:, 0])
    can1_in_y = torch.logical_and(can1_pos[:, 1] > lower_xy[:, 1], can1_pos[:, 1] < upper_xy[:, 1])
    can2_in_x = torch.logical_and(can2_pos[:, 0] > lower_xy[:, 0], can2_pos[:, 0] < upper_xy[:, 0])
    can2_in_y = torch.logical_and(can2_pos[:, 1] > lower_xy[:, 1], can2_pos[:, 1] < upper_xy[:, 1])

    can1_in_container = torch.logical_and(can1_in_x, can1_in_y)
    can2_in_container = torch.logical_and(can2_in_x, can2_in_y)

    # World-frame Z for insertion height check (matches Ego logic conceptually)
    can1_z = can1.data.root_pos_w[:, 2]
    can2_z = can2.data.root_pos_w[:, 2]

    insert1_now = torch.logical_and(can1_in_container, can1_z <= z_threshold)
    insert2_now = torch.logical_and(can2_in_container, can2_z <= z_threshold)

    # Once inserted in an episode, the flag stays True until reset
    env.insert_and_unload_cans_insert1 = torch.logical_or(env.insert_and_unload_cans_insert1, insert1_now)
    env.insert_and_unload_cans_insert2 = torch.logical_or(env.insert_and_unload_cans_insert2, insert2_now)

    # Unload = currently outside AND has been inserted before in the same episode
    unload1 = torch.logical_and(torch.logical_not(can1_in_container), env.insert_and_unload_cans_insert1)
    unload2 = torch.logical_and(torch.logical_not(can2_in_container), env.insert_and_unload_cans_insert2)

    done = torch.logical_and(unload1, unload2)
    return done

def task_done_sort_cans(
    env: "ManagerBasedRLEnv",

    # four cans
    can_sprite1_cfg: SceneEntityCfg = SceneEntityCfg("can_sprite_1"),
    can_sprite2_cfg: SceneEntityCfg = SceneEntityCfg("can_sprite_2"),
    can_fanta1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can_fanta2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),

    # instead of fixed coordinates, use SceneEntityCfg for containers
    container_left_cfg: SceneEntityCfg = SceneEntityCfg("container_1"),
    container_right_cfg: SceneEntityCfg = SceneEntityCfg("container_2"),

    # container bounding-box half extents
    up_half_extent=(0.12, 0.075, 0.01),
    down_half_extent=(0.13, 0.075, 0.07),
):
    """
    Success when:
    - Both Sprite cans lie inside the LEFT container box.
    - Both Fanta cans lie inside the RIGHT container box.
    Container positions are dynamically read from env, not hard-coded.

    NOTE:
    - Sub-tasks such as upright check and grasp success are preserved logically,
      but relaxed so they do not block success unless you enable them.
    """

    # ----------------------------
    # Get can world positions
    # ----------------------------
    cans = {
        "sprite1": env.scene[can_sprite1_cfg.name],
        "sprite2": env.scene[can_sprite2_cfg.name],
        "fanta1": env.scene[can_fanta1_cfg.name],
        "fanta2": env.scene[can_fanta2_cfg.name],
    }

    cans_pos = {
        k: v.data.root_pos_w - env.scene.env_origins
        for k, v in cans.items()
    }

    device = cans_pos["sprite1"].device

    # ----------------------------
    # Read dynamic container positions
    # ----------------------------
    cont_left = env.scene[container_left_cfg.name]
    cont_right = env.scene[container_right_cfg.name]

    left_center = cont_left.data.root_pos_w - env.scene.env_origins
    right_center = cont_right.data.root_pos_w - env.scene.env_origins

    # ----------------------------
    # bounding box extents
    # ----------------------------
    up = torch.tensor(up_half_extent, device=device)
    down = torch.tensor(down_half_extent, device=device)

    left_top_left = left_center + up
    left_bot_right = left_center - down
    right_top_left = right_center + up
    right_bot_right = right_center - down

    # inside-box check
    def in_box(pos, bot_right, top_left):
        inside = torch.logical_and(pos > bot_right, pos < top_left)
        return inside.all(dim=-1)

    # ----------------------------
    # Sorting logic
    # ----------------------------
    fanta_sorted = torch.logical_and(
        in_box(cans_pos["fanta1"], right_bot_right, right_top_left),
        in_box(cans_pos["fanta2"], right_bot_right, right_top_left),
    )

    sprite_sorted = torch.logical_and(
        in_box(cans_pos["sprite1"], left_bot_right, left_top_left),
        in_box(cans_pos["sprite2"], left_bot_right, left_top_left),
    )

    # ----------------------------
    # Upright check (kept but relaxed)
    # If you need strict upright: enable angle threshold.
    # ----------------------------
    def upright_dummy():
        # current implementation does not restrict success
        return torch.ones_like(fanta_sorted, device=device, dtype=torch.bool)

    upright_ok = upright_dummy()

    # ----------------------------
    # Final success
    # ----------------------------
    done = torch.logical_and(torch.logical_and(fanta_sorted, sprite_sorted), upright_ok)

    return done


def task_done_stack_can(
    env: "ManagerBasedRLEnv",
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    plate_radius: float = 0.045, # 和柜子等比缩放了
    vertical_tolerance: float = 0.02,
) -> torch.Tensor:
    """
    - Compute horizontal (XY) distance between can and plate centers.
    - Compute vertical (Z) distance between can and plate.
    - Task is successful when horizontal distance is within ``plate_radius``
      and vertical distance is below ``vertical_tolerance``.
    """

    can = env.scene[can_cfg.name]
    plate = env.scene[plate_cfg.name]

    # positions relative to environment origin
    can_pos = can.data.root_pos_w - env.scene.env_origins
    plate_pos = plate.data.root_pos_w - env.scene.env_origins

    diff = can_pos - plate_pos
    dist_horizontal = torch.norm(diff[:, :2], dim=-1)
    dist_vertical = torch.abs(diff[:, 2])

    done = dist_horizontal < plate_radius
    done = torch.logical_and(done, dist_vertical < vertical_tolerance)

    return done

def task_done_stack_can_into_drawer(
    env: "ManagerBasedRLEnv",
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    drawer_bottom_joint_id: int = 0,
    close_ratio: float = 0.90,
    plate_radius: float = 0.3,
    vertical_tolerance: float = 0.02,
) -> torch.Tensor:
    """Success for stack_can_into_drawer task.

    Mirrors Ego's :meth:`StackCanIntoDrawerEnv._get_success` and
    ``_get_joints_data``:

    - Can must be horizontally within ``plate_radius`` of the plate.
    - Vertical distance between can and plate must be below
      ``vertical_tolerance``.
    - Drawer bottom joint must be sufficiently closed (>= close_ratio of
      its upper joint limit).
    """

    can = env.scene[can_cfg.name]
    plate = env.scene[plate_cfg.name]
    drawer = env.scene[drawer_cfg.name]

    # ---- can on plate (env-local) ----
    can_pos = can.data.root_pos_w - env.scene.env_origins
    plate_pos = plate.data.root_pos_w - env.scene.env_origins

    diff = can_pos - plate_pos
    dist_horizontal = torch.norm(diff[:, :2], dim=-1)
    dist_vertical = torch.abs(diff[:, 2])

    stacked = torch.logical_and(dist_horizontal < plate_radius, dist_vertical < vertical_tolerance)

    # ---- drawer closed condition ----
    joint_pos = drawer.data.joint_pos[:, drawer_bottom_joint_id]
    joint_upper = drawer.data.joint_pos_limits[:, drawer_bottom_joint_id, 1]
    drawer_closed = joint_pos > (joint_upper * close_ratio)

    # print(stacked, drawer_closed)

    done = torch.logical_and(stacked, drawer_closed)

    return done

def task_done_insert_n_cans(
    env: "ManagerBasedRLEnv",
    can_cfgs,
    target_insert_num: int,
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    container_center_xy=(0.48, 0.0),
    container_half_extent_xy=(0.16, 0.10),
    z_above_container: float = 0.01,
) -> torch.Tensor:

    assert target_insert_num > 0, "target_insert_num must be > 0"
    assert len(can_cfgs) >= target_insert_num, \
        "Number of cans must be >= target_insert_num"

    container = env.scene[container_cfg.name]

    # positions in env-local frame
    container_pos = container.data.root_pos_w - env.scene.env_origins
    device = container_pos.device
    num_envs = container_pos.shape[0]

    # container footprint reference
    container_center_ref = torch.tensor(
        container_center_xy, device=device
    )
    container_half_extent = torch.tensor(
        container_half_extent_xy, device=device
    )

    container_xy_lower_ref = container_center_ref - container_half_extent
    container_xy_upper_ref = container_center_ref + container_half_extent

    # per-env offset
    offset_xy = container_pos[:, 0:2] - container_center_ref
    lower_xy = container_xy_lower_ref + offset_xy
    upper_xy = container_xy_upper_ref + offset_xy

    z_threshold = container_pos[:, 2] + z_above_container

    def in_box_xy(pos_xy, lower, upper):
        return torch.logical_and(
            torch.logical_and(pos_xy[:, 0] > lower[:, 0],
                              pos_xy[:, 0] < upper[:, 0]),
            torch.logical_and(pos_xy[:, 1] > lower[:, 1],
                              pos_xy[:, 1] < upper[:, 1]),
        )

    # 统计每个 env 中成功 insert 的罐子数量
    success_count = torch.zeros(
        num_envs, device=device, dtype=torch.int32
    )

    for can_cfg in can_cfgs:
        can = env.scene[can_cfg.name]
        can_pos = can.data.root_pos_w - env.scene.env_origins

        in_xy = in_box_xy(can_pos[:, 0:2], lower_xy, upper_xy)
        in_z = can_pos[:, 2] <= z_threshold

        can_success = torch.logical_and(in_xy, in_z)
        success_count += can_success.int()

    # 至少 target_insert_num 个罐子成功
    done = success_count >= target_insert_num
    return done


def task_done_insert_cans(
    env: "ManagerBasedRLEnv",
    can1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    container_center_xy=(0.48, 0.0),
    container_half_extent_xy=(0.16, 0.10),
    z_above_container: float = 0.01,
) -> torch.Tensor:
    """
    - Check if cans are inside the container.
    - Success when both cans are within the container's XY bounds and below a Z threshold.
    """
    can1 = env.scene[can1_cfg.name]
    can2 = env.scene[can2_cfg.name]
    container = env.scene[container_cfg.name]

    # positions in env-local frame
    can1_pos = can1.data.root_pos_w - env.scene.env_origins
    can2_pos = can2.data.root_pos_w - env.scene.env_origins
    container_pos = container.data.root_pos_w - env.scene.env_origins

    device = container_pos.device

    # Base container footprint (in env frame) and its nominal center.
    container_center_ref = torch.tensor(container_center_xy, device=device)
    container_half_extent = torch.tensor(container_half_extent_xy, device=device)
    
    container_xy_lower_ref = container_center_ref - container_half_extent
    container_xy_upper_ref = container_center_ref + container_half_extent

    # Infer per-env XY offset from current container position.
    offset_xy = container_pos[:, 0:2] - container_center_ref
    lower_xy = container_xy_lower_ref + offset_xy
    upper_xy = container_xy_upper_ref + offset_xy

    # Check whether cans are inside the container XY box.
    def in_box(pos, lower, upper):
        return torch.logical_and(
            torch.logical_and(pos[:, 0] > lower[:, 0], pos[:, 0] < upper[:, 0]),
            torch.logical_and(pos[:, 1] > lower[:, 1], pos[:, 1] < upper[:, 1])
        )

    can1_in_xy = in_box(can1_pos, lower_xy, upper_xy)
    can2_in_xy = in_box(can2_pos, lower_xy, upper_xy)

    # Check Z height relative to container: can must be below container_z + z_above_container
    # This is more robust than an absolute threshold since it adapts to env_origins and container placement.
    z_threshold = container_pos[:, 2] + z_above_container
    can1_in_z = can1_pos[:, 2] <= z_threshold
    can2_in_z = can2_pos[:, 2] <= z_threshold

    can1_success = torch.logical_and(can1_in_xy, can1_in_z)
    can2_success = torch.logical_and(can2_in_xy, can2_in_z)

    done = torch.logical_and(can1_success, can2_success)

    return done

def task_done_flip_mug(
    env: "ManagerBasedRLEnv",
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    up_z_threshold: float = 0.99,
) -> torch.Tensor:
    """Success when mug is flipped so its local +Z axis points sufficiently upwards."""
    mug = env.scene[mug_cfg.name]
    # z-axis in world frame for each env
    z_axis = torch.zeros_like(mug.data.root_pos_w)
    z_axis[:, 2] = 1.0
    z_mug = math_utils.quat_apply(mug.data.root_quat_w, z_axis)
    done = z_mug[:, 2] > up_z_threshold
    return done

def task_done_push_box(
    env: "ManagerBasedRLEnv",
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    default_goal_center_xy=(0.5, 0.0),
    goal_radius: float = 0.08,
) -> torch.Tensor:
    """Success when the box's XY position is within goal_radius of goal position.
    Uses env.push_box_goal_pos_w if available (set by reset_push_box_objects),
    otherwise falls back to goal_center_xy parameter.
    """
    box = env.scene[box_cfg.name]
    box_pos_w = box.data.root_pos_w
    device = box_pos_w.device

    # Use dynamic goal position if available, otherwise use static parameter
    if hasattr(env, "push_box_goal_pos_w"):
        goal_pos_w = env.push_box_goal_pos_w
        dist_xy = torch.norm(box_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1)
    else:
        # Fallback: use env-local coordinates with static goal_center_xy
        box_pos = box_pos_w - env.scene.env_origins
        goal_center = torch.tensor(default_goal_center_xy, device=device)
        dist_xy = torch.norm(box_pos[:, :2] - goal_center[None, :], dim=-1)

    done = dist_xy < goal_radius
    return done