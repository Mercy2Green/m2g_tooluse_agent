# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_laptop_open(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,  # 需要重置的环境 ID 列表（张量形式）
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),  # 指定笔记本在场景中的配置，默认名称为 "laptop"
    randomize: bool = True,  # 是否对笔记本在 x/y 方向的位置进行随机扰动
    randomize_idx: int = -1,  # 如果 >= 0，使用固定网格上的确定位置；< 0 时使用连续随机采样
    randomize_range: float = 1.0,  # 随机扰动的范围缩放系数
) -> None:
    """
    重置笔记本（laptop）的状态，与 EgoVLA 中 OpenLaptopEnv._reset_idx 的逻辑类似。

    主要功能：
    - 将笔记本的根状态（位置 + 姿态）放在对应 env 的原点附近；
    - 可选地对笔记本在 x/y 平面上的位置进行随机扰动（连续随机或规则网格）；
    - 将笔记本的关节位置重置为默认值（并裁剪到软关节限制范围内），关节速度重置为 0。
    """
    # 从场景中根据配置名称获取笔记本这个关节体（articulation）对象
    laptop = env.scene[laptop_cfg.name]

    # ===== 根姿态重置部分 =====
    # 从默认根状态中取出对应 env_ids 的根状态，并做一次拷贝，避免直接修改底层缓存
    laptop_root_state = laptop.data.default_root_state[env_ids].clone()
    # 将根位置平移到对应环境的原点位置处（每个 env 的原点可能不同）
    laptop_root_state[:, 0:3] += env.scene.env_origins[env_ids, :]

    # ===== 位置随机化部分（可选）=====
    if randomize:
        # randomize_idx < 0：使用连续的随机扰动
        if randomize_idx < 0:
            # 在 [-0.1, 0.1] * randomize_range 范围内随机生成 x/y 偏移
            # rand_xy 的形状为 [num_envs, 2]，分别对应 x 和 y 的偏移量
            rand_xy = (0.20 * randomize_range) * torch.rand((len(env_ids), 2), device=laptop.device) - (
                0.1 * randomize_range
            )
            # 将随机偏移加到根状态的 x/y 坐标上
            laptop_root_state[:, 0:2] += rand_xy
        else:
            # randomize_idx >= 0：使用 100x100 的规则网格，在 [-0.1, 0.1] 范围内取一个确定位置
            # 计算网格列索引（相当于 x 方向）
            column_idx = randomize_idx // 100
            # 计算网格行索引（相当于 y 方向）
            row_idx = randomize_idx % 100
            # 根据行列索引映射到 [-0.1, 0.1] 区间中的具体坐标（y 方向）
            laptop_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
            # 根据行列索引映射到 [-0.1, 0.1] 区间中的具体坐标（x 方向）
            laptop_root_state[:, 0] += 0.1 - (0.2 / 99) * column_idx

    # 将更新后的根状态一次性写回到仿真中，应用到所有指定 env_ids 的笔记本
    laptop.write_root_state_to_sim(laptop_root_state, env_ids=env_ids)
    laptop_root_state = laptop.data.default_root_state[env_ids].clone()
    print("default_root_state quat:", laptop_root_state[0, 3:7])
    # ===== 关节状态重置部分 =====
    # 取出这些环境中笔记本的默认关节位置（例如：盖子的初始开合角度等）
    joint_pos = laptop.data.default_joint_pos[env_ids]
    # 将关节位置裁剪到软关节限制范围内，防止越界导致不稳定
    joint_pos = torch.clamp(
        joint_pos,
        laptop.data.soft_joint_pos_limits[0, :, 0],  # 每个关节的下限
        laptop.data.soft_joint_pos_limits[0, :, 1],  # 每个关节的上限
    )
    # 创建与 joint_pos 相同形状的零张量，作为初始关节速度（所有关节速度设为 0）
    joint_vel = torch.zeros_like(joint_pos)

    # 将关节位置和速度写入仿真，用于重置笔记本的关节状态
    laptop.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    print("reset laptop")

def reset_drawer_closed(env, env_ids, asset_cfg: SceneEntityCfg):
    drawer = env.scene[asset_cfg.name]

    bottom_joint_idx = 0
    top_joint_idx = 1

    joint_limits = drawer.data.joint_limits
    bottom_close = joint_limits[env_ids, bottom_joint_idx, 1]
    top_close = joint_limits[env_ids, top_joint_idx, 0]

    # 只取需要 reset 的 env
    joint_pos = drawer.data.joint_pos[env_ids].clone()
    joint_vel = drawer.data.joint_vel[env_ids].clone()

    joint_pos[:, bottom_joint_idx] = bottom_close
    joint_pos[:, top_joint_idx] = top_close

    drawer.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    return {}




def reset_object_poses_nut_pour(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
):
    """Reset the asset root states to a random position and orientation uniformly within the given ranges.

    Args:
        env: The RL environment instance.
        env_ids: The environment IDs to reset the object poses for.
        sorting_beaker_cfg: The configuration for the sorting beaker asset.
        factory_nut_cfg: The configuration for the factory nut asset.
        sorting_bowl_cfg: The configuration for the sorting bowl asset.
        sorting_scale_cfg: The configuration for the sorting scale asset.
        pose_range: The dictionary of pose ranges for the objects. Keys are
                    ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.
    """
    # extract the used quantities (to enable type-hinting)
    sorting_beaker = env.scene[sorting_beaker_cfg.name]
    factory_nut = env.scene[factory_nut_cfg.name]
    sorting_bowl = env.scene[sorting_bowl_cfg.name]
    sorting_scale = env.scene[sorting_scale_cfg.name]

    # get default root state
    sorting_beaker_root_states = sorting_beaker.data.default_root_state[env_ids].clone()
    factory_nut_root_states = factory_nut.data.default_root_state[env_ids].clone()
    sorting_bowl_root_states = sorting_bowl.data.default_root_state[env_ids].clone()
    sorting_scale_root_states = sorting_scale.data.default_root_state[env_ids].clone()

    # get pose ranges
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=sorting_beaker.device)

    # randomize sorting beaker and factory nut together
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_beaker = (
        sorting_beaker_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    )
    positions_factory_nut = factory_nut_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_beaker = math_utils.quat_mul(sorting_beaker_root_states[:, 3:7], orientations_delta)
    orientations_factory_nut = math_utils.quat_mul(factory_nut_root_states[:, 3:7], orientations_delta)

    # randomize sorting bowl
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_bowl = sorting_bowl_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_bowl = math_utils.quat_mul(sorting_bowl_root_states[:, 3:7], orientations_delta)

    # randomize scorting scale
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_scale = sorting_scale_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_scale = math_utils.quat_mul(sorting_scale_root_states[:, 3:7], orientations_delta)

    # set into the physics simulation
    sorting_beaker.write_root_pose_to_sim(
        torch.cat([positions_sorting_beaker, orientations_sorting_beaker], dim=-1), env_ids=env_ids
    )
    factory_nut.write_root_pose_to_sim(
        torch.cat([positions_factory_nut, orientations_factory_nut], dim=-1), env_ids=env_ids
    )
    sorting_bowl.write_root_pose_to_sim(
        torch.cat([positions_sorting_bowl, orientations_sorting_bowl], dim=-1), env_ids=env_ids
    )
    sorting_scale.write_root_pose_to_sim(
        torch.cat([positions_sorting_scale, orientations_sorting_scale], dim=-1), env_ids=env_ids
    )




def reset_unload_cans_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    #randomize: bool = False,
    #randomize_idx: int = -1,
    #randomize_range: float = 1.0,
    can1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
) -> None:
    can1 = env.scene[can1_cfg.name]
    can2 = env.scene[can2_cfg.name]
    container = env.scene[container_cfg.name]

    # base default root states (env-local) for the selected envs
    can1_root_state = can1.data.default_root_state[env_ids].clone()
    can2_root_state = can2.data.default_root_state[env_ids].clone()
    container_root_state = container.data.default_root_state[env_ids].clone()

    # move to world frame around env origins
    can1_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    can2_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    container_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    # 不进行随机位置初始化
    # sample shared offset in XY plane
    #offset_xy = torch.zeros(2, device=can1.device)
    # if randomize:
    #     if randomize_idx < 0:
    #         offset_xy = (randomize_range * 0.2) * torch.rand((2,), device=can1.device) - (
    #             randomize_range * 0.1
    #         )
    #     else:
    #         column_idx = randomize_idx // 100
    #         row_idx = randomize_idx % 100
    #         offset_xy[1] += 0.1 - (0.2 / 99.0) * row_idx
    #         offset_xy[0] += 0.05 - (0.1 / 99.0) * column_idx

    #can1_root_state[:, 0:2] += offset_xy
    #can2_root_state[:, 0:2] += offset_xy
    #container_root_state[:, 0:2] += offset_xy

    can1.write_root_state_to_sim(can1_root_state, env_ids=env_ids)
    can2.write_root_state_to_sim(can2_root_state, env_ids=env_ids)
    container.write_root_state_to_sim(container_root_state, env_ids=env_ids)


def reset_sort_cans_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    # randomize: bool = True,
    # randomize_idx: int = -1,
    # randomize_range: float = 1.0,
    can_sprite1_cfg: SceneEntityCfg = SceneEntityCfg("can_sprite_1"),
    can_sprite2_cfg: SceneEntityCfg = SceneEntityCfg("can_sprite_2"),
    can_fanta1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can_fanta2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
) -> None:

    can_sprite_1 = env.scene[can_sprite1_cfg.name]
    can_sprite_2 = env.scene[can_sprite2_cfg.name]
    can_fanta_1 = env.scene[can_fanta1_cfg.name]
    can_fanta_2 = env.scene[can_fanta2_cfg.name]

    cans = [can_sprite_1, can_sprite_2, can_fanta_1, can_fanta_2]

    for can_idx, can in enumerate(cans):
        root_state = can.data.default_root_state[env_ids].clone()

        # if randomize:
        #     if randomize_idx < 0:
        #         # uniform noise in XY, scaled by randomize_range
        #         noise_xy = (randomize_range * 0.12) * torch.rand(
        #             (len(env_ids), 2), device=can.device
        #         ) - (0.06 * randomize_range)
        #         root_state[:, 0:2] += noise_xy
        #     else:
        #         # deterministic grid layout, same logic as Ego:
        #         # randomize_idx encodes left/right grid indices.
        #         left_cans_idx = randomize_idx // 100
        #         left_cans_column_idx = left_cans_idx // 10
        #         left_cans_row_idx = left_cans_idx % 10

        #         right_cans_idx = randomize_idx % 100
        #         right_cans_column_idx = right_cans_idx // 10
        #         right_cans_row_idx = right_cans_idx % 10

        #         if can_idx in (0, 2):
        #             # can_sprite_1 and can_fanta_1 (right side)
        #             root_state[:, 1] += 0.06 - (0.12 / 9.0) * right_cans_row_idx
        #             root_state[:, 0] += 0.06 - (0.12 / 9.0) * right_cans_column_idx
        #         else:
        #             # can_sprite_2 and can_fanta_2 (left side)
        #             root_state[:, 1] += 0.06 - (0.12 / 9.0) * left_cans_row_idx
        #             root_state[:, 0] += 0.06 - (0.12 / 9.0) * left_cans_column_idx

        # move into world frame around env origins
        root_state[:, 0:3] += env.scene.env_origins[env_ids, :]
        can.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_stack_can_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    # randomize: bool = True,
    # randomize_idx: int = -1,
    # randomize_range: float = 1.0,
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> None:
    """
    - Start from default root states for can and plate.
    - If ``randomize_idx < 0``, independently randomize XY for both within
      a square region scaled by ``randomize_range``.
    - If ``randomize_idx >= 0``, use a deterministic 100x100 grid position
      shared by can and plate.
    - Finally, shift both by the environment origins and write to sim.
    """

    can = env.scene[can_cfg.name]
    plate = env.scene[plate_cfg.name]

    can_root_state = can.data.default_root_state[env_ids].clone()
    plate_root_state = plate.data.default_root_state[env_ids].clone()

    # if randomize:
    #     if randomize_idx < 0:
    #         # uniform XY noise in [-0.1, 0.1] scaled by randomize_range
    #         plate_noise_xy = (randomize_range * 0.2) * torch.rand(
    #             (len(env_ids), 2), device=can.device
    #         ) - (0.1 * randomize_range)
    #         can_noise_xy = (randomize_range * 0.2) * torch.rand(
    #             (len(env_ids), 2), device=can.device
    #         ) - (0.1 * randomize_range)
    #         plate_root_state[:, 0:2] += plate_noise_xy
    #         can_root_state[:, 0:2] += can_noise_xy
    #     else:
    #         # deterministic 100x100 grid
    #         column_idx = randomize_idx // 100
    #         row_idx = randomize_idx % 100
    #         plate_root_state[:, 1] += 0.1 - (0.2 / 99.0) * row_idx
    #         plate_root_state[:, 0] += 0.1 - (0.2 / 99.0) * column_idx
    #         can_root_state[:, 1] += 0.1 - (0.2 / 99.0) * row_idx
    #         can_root_state[:, 0] += 0.1 - (0.2 / 99.0) * column_idx

    # move into world frame around env origins
    can_root_state[:, 0:3] += env.scene.env_origins[env_ids, :]
    plate_root_state[:, 0:3] += env.scene.env_origins[env_ids, :]

    can.write_root_state_to_sim(can_root_state, env_ids=env_ids)
    plate.write_root_state_to_sim(plate_root_state, env_ids=env_ids)

def reset_stack_can_into_drawer_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    # randomize: bool = True,
    # randomize_idx: int = -1,
    # randomize_range: float = 1.0,
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    drawer_top_joint_name: str = "top_joint",
    drawer_bottom_joint_name: str = "bottom_joint",
    drawer_init_state: str = "open",
) -> None:
    """Reset can, plate and drawer for the stack_can_into_drawer task.
    - Reset drawer joints to an open or closed configuration.
    - Optionally randomize XY poses of drawer, can and plate using
      either continuous noise or a deterministic grid index.
    - Shift all roots by ``env.scene.env_origins`` and write to sim.
    """

    can = env.scene[can_cfg.name]
    plate = env.scene[plate_cfg.name]
    drawer = env.scene[drawer_cfg.name]

    # resolve joint indices
    drawer_top_joint_id = drawer.find_joints(drawer_top_joint_name)[0][0]
    drawer_bottom_joint_id = drawer.find_joints(drawer_bottom_joint_name)[0][0]

    # ----- reset drawer joints -----
    joint_pos = drawer.data.default_joint_pos[env_ids].clone()

    bottom_joint_bound = 1 if drawer_init_state == "close" else 0
    joint_limits = drawer.data.joint_limits
    print(joint_pos)
    joint_pos[:, drawer_top_joint_id] = joint_limits[env_ids, drawer_top_joint_id, 1].squeeze()
    joint_pos[:, drawer_bottom_joint_id] = joint_limits[env_ids, drawer_bottom_joint_id, bottom_joint_bound].squeeze()
    print(joint_pos)
    joint_pos = torch.clamp(
        joint_pos,
        drawer.data.soft_joint_pos_limits[0, :, 0],
        drawer.data.soft_joint_pos_limits[0, :, 1],
    )
    print(joint_pos)
    joint_vel = torch.zeros_like(joint_pos)
    drawer.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # ----- reset root poses -----
    can_root_state = can.data.default_root_state[env_ids].clone()
    plate_root_state = plate.data.default_root_state[env_ids].clone()
    drawer_root_state = drawer.data.default_root_state[env_ids].clone()

    # move into world frame around env origins
    can_root_state[:, 0:3] += env.scene.env_origins[env_ids, :]
    plate_root_state[:, 0:3] += env.scene.env_origins[env_ids, :]
    drawer_root_state[:, 0:3] += env.scene.env_origins[env_ids, :]

    # if randomize:
    #     if randomize_idx < 0:
    #         # shared noise for drawer and can in XY
    #         drawer_noise = (randomize_range * 0.1) * torch.rand(
    #             (len(env_ids), 2), device=drawer.device
    #         ) - (0.05 * randomize_range)
    #         drawer_root_state[:, 0:2] += drawer_noise
    #         can_root_state[:, 0:2] += drawer_noise

    #         # independent noise for plate
    #         plate_noise = (randomize_range * 0.1) * torch.rand(
    #             (len(env_ids), 2), device=plate.device
    #         ) - (0.05 * randomize_range)
    #         plate_root_state[:, 0:2] += plate_noise
    #     else:
    #         column_idx = randomize_idx // 100
    #         row_idx = randomize_idx % 100

    #         drawer_root_state[:, 1] += 0.1 - (0.2 / 99.0) * row_idx
    #         plate_root_state[:, 1] += 0.1 - (0.2 / 99.0) * row_idx
    #         can_root_state[:, 1] += 0.1 - (0.2 / 99.0) * row_idx
    #         can_root_state[:, 0] += 0.05 - (0.1 / 99.0) * column_idx

    drawer.write_root_state_to_sim(drawer_root_state, env_ids=env_ids)
    can.write_root_state_to_sim(can_root_state, env_ids=env_ids)
    plate.write_root_state_to_sim(plate_root_state, env_ids=env_ids)


def reset_insert_cans_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    # randomize: bool = False,
    # randomize_idx: int = -1,
    # randomize_range: float = 1.0,
    can1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
) -> None:
    can1 = env.scene[can1_cfg.name]
    can2 = env.scene[can2_cfg.name]
    container = env.scene[container_cfg.name]

    # base default root states (env-local) for the selected envs
    can1_root_state = can1.data.default_root_state[env_ids].clone()
    can2_root_state = can2.data.default_root_state[env_ids].clone()
    container_root_state = container.data.default_root_state[env_ids].clone()

    # move to world frame around env origins
    can1_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    can2_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    container_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    # Deterministic reset (no randomization for now)
    # If randomization is needed, copy logic from reset_unload_cans_objects or similar.

    can1.write_root_state_to_sim(can1_root_state, env_ids=env_ids)
    can2.write_root_state_to_sim(can2_root_state, env_ids=env_ids)
    container.write_root_state_to_sim(container_root_state, env_ids=env_ids)


def reset_insert_and_unload_cans_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    can1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
) -> None:
    """Reset cans and container for insert_and_unload_cans task and clear per-episode flags.

    Layout is the same as insert_cans / unload_cans: deterministic positions around env origins.
    """

    can1 = env.scene[can1_cfg.name]
    can2 = env.scene[can2_cfg.name]
    container = env.scene[container_cfg.name]

    can1_root_state = can1.data.default_root_state[env_ids].clone()
    can2_root_state = can2.data.default_root_state[env_ids].clone()
    container_root_state = container.data.default_root_state[env_ids].clone()

    can1_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    can2_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    container_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    can1.write_root_state_to_sim(can1_root_state, env_ids=env_ids)
    can2.write_root_state_to_sim(can2_root_state, env_ids=env_ids)
    container.write_root_state_to_sim(container_root_state, env_ids=env_ids)

    # Reset episode-local insert flags used by task_done_insert_and_unload_cans
    device = env.scene.env_origins.device
    num_envs_total = env.scene.env_origins.shape[0]
    if (not hasattr(env, "insert_and_unload_cans_insert1")) or env.insert_and_unload_cans_insert1.shape[0] != num_envs_total:
        env.insert_and_unload_cans_insert1 = torch.zeros(num_envs_total, dtype=torch.bool, device=device)
        env.insert_and_unload_cans_insert2 = torch.zeros(num_envs_total, dtype=torch.bool, device=device)
    env.insert_and_unload_cans_insert1[env_ids] = False
    env.insert_and_unload_cans_insert2[env_ids] = False


def reset_pour_balls_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    bottle_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    ball1_cfg: SceneEntityCfg = SceneEntityCfg("ball1"),
    ball2_cfg: SceneEntityCfg = SceneEntityCfg("ball2"),
    ball3_cfg: SceneEntityCfg = SceneEntityCfg("ball3"),
    ball4_cfg: SceneEntityCfg = SceneEntityCfg("ball4"),
    ball5_cfg: SceneEntityCfg = SceneEntityCfg("ball5"),
    ball6_cfg: SceneEntityCfg = SceneEntityCfg("ball6"),
    ball7_cfg: SceneEntityCfg = SceneEntityCfg("ball7"),
    ball8_cfg: SceneEntityCfg = SceneEntityCfg("ball8"),
) -> None:
    """Reset bottle, bowl and balls for pour_balls task and clear per-episode flags."""

    bottle = env.scene[bottle_cfg.name]
    bowl = env.scene[bowl_cfg.name]
    balls_cfg = [ball1_cfg, ball2_cfg, ball3_cfg, ball4_cfg, ball5_cfg, ball6_cfg, ball7_cfg, ball8_cfg]
    balls = [env.scene[cfg.name] for cfg in balls_cfg]

    bottle_root_state = bottle.data.default_root_state[env_ids].clone()
    bowl_root_state = bowl.data.default_root_state[env_ids].clone()
    balls_root_states = [ball.data.default_root_state[env_ids].clone() for ball in balls]

    bottle_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    bowl_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    for root_state in balls_root_states:
        root_state[:, 0:3] += env.scene.env_origins[env_ids]

    bottle.write_root_state_to_sim(bottle_root_state, env_ids=env_ids)
    bowl.write_root_state_to_sim(bowl_root_state, env_ids=env_ids)
    for ball, root_state in zip(balls, balls_root_states):
        ball.write_root_state_to_sim(root_state, env_ids=env_ids)

    # Reset episode-local pour flags and counts used by task_done_pour_balls
    device = env.scene.env_origins.device
    num_envs_total = env.scene.env_origins.shape[0]
    num_balls = len(balls)
    if (not hasattr(env, "pour_balls_pass")) or env.pour_balls_pass.shape != (num_envs_total, num_balls):
        env.pour_balls_pass = torch.zeros((num_envs_total, num_balls), dtype=torch.bool, device=device)
        env.pour_balls_count = torch.zeros(num_envs_total, dtype=torch.int32, device=device)
    env.pour_balls_pass[env_ids, :] = False
    env.pour_balls_count[env_ids] = 0


def reset_flip_mug_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    randomize: bool = False,
    randomize_idx: int = -1,
    randomize_range: float = 1.0,
) -> None:
    """Reset mug root pose for flip_mug task (Ego-style randomization optional, disabled by default)."""

    mug = env.scene[mug_cfg.name]
    mug_root_state = mug.data.default_root_state[env_ids].clone()

    # if randomize:
    #     if randomize_idx < 0:
    #         noise_xy = (0.2 * randomize_range) * torch.rand((len(env_ids), 2), device=mug.device) - (
    #             0.1 * randomize_range
    #         )
    #         mug_root_state[:, 0:2] += noise_xy
    #     else:
    #         column_idx = randomize_idx // 100
    #         row_idx = randomize_idx % 100
    #         mug_root_state[:, 1] += 0.1 - (0.2 / 99.0) * row_idx
    #         mug_root_state[:, 0] += 0.1 - (0.2 / 99.0) * column_idx

    mug_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    mug.write_root_state_to_sim(mug_root_state, env_ids=env_ids)


def reset_push_box_objects(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    default_goal_pos: tuple[float, float, float] = (0.5, 0.0, 0.687),
) -> None:
    """Reset box root pose and goal marker for push_box task.

    Current version keeps a deterministic layout (no randomization of box or goal).
    The goal marker is lazily created on first reset and updated each reset.
    """
    from isaaclab.markers.visualization_markers import VisualizationMarkers

    num_envs = env.scene.num_envs
    device = env.device

    # Lazily create goal_pos_w buffer and goal_marker
    if not hasattr(env, "push_box_goal_pos_w"):
        env.push_box_goal_pos_w = torch.zeros((num_envs, 3), device=device)
        # Create goal marker from env cfg (goal_cfg is on env.cfg, not env.cfg.scene)
        env.push_box_goal_marker = VisualizationMarkers(env.cfg.goal_cfg)

    # Reset box
    box = env.scene[box_cfg.name]
    box_root_state = box.data.default_root_state[env_ids].clone()
    box_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    box.write_root_state_to_sim(box_root_state, env_ids=env_ids)

    # Reset goal position (deterministic, no randomization for now)
    goal_pos = torch.tensor(default_goal_pos, device=device)
    env.push_box_goal_pos_w[env_ids] = goal_pos + env.scene.env_origins[env_ids]

    # Update goal marker visualization
    env.push_box_goal_marker.visualize(env.push_box_goal_pos_w)