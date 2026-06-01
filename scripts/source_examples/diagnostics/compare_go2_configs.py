#!/usr/bin/env python3
"""Print a focused config diff between official IsaacLab Go2 and local Go2+Piper.

Run from the repository root through the IsaacLab Python environment, for example:

    cd $M2G_TOOLUSE_ROOT
    conda activate isaacsim5
    $ISAACLAB_PATH/isaaclab.sh -p scripts/source_examples/diagnostics/compare_go2_configs.py
"""

from __future__ import annotations

import pprint
import os
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
M2G_ROOT = THIS_FILE.parents[3]
WORKSPACE_ROOT = M2G_ROOT.parent
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", WORKSPACE_ROOT / "IsaacLab"))

for path in (
    M2G_ROOT,
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
    ISAACLAB_ROOT / "source" / "isaaclab_rl",
):
    sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher  # noqa: E402


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def value(obj, dotted: str, default="<missing>"):
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def dump(title: str, data) -> None:
    print(f"\n## {title}")
    pprint.pp(data, sort_dicts=False, width=120)


def actuator_summary(cfg) -> dict:
    out = {}
    for name, actuator in value(cfg, "actuators", {}).items():
        out[name] = {
            "type": type(actuator).__name__,
            "joint_names_expr": value(actuator, "joint_names_expr"),
            "effort_limit": value(actuator, "effort_limit"),
            "effort_limit_sim": value(actuator, "effort_limit_sim"),
            "saturation_effort": value(actuator, "saturation_effort"),
            "velocity_limit": value(actuator, "velocity_limit"),
            "velocity_limit_sim": value(actuator, "velocity_limit_sim"),
            "stiffness": value(actuator, "stiffness"),
            "damping": value(actuator, "damping"),
            "friction": value(actuator, "friction"),
        }
    return out


def reward_summary(rewards) -> dict:
    names = (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "flat_orientation_l2",
        "feet_air_time",
        "dof_torques_l2",
        "dof_acc_l2",
        "hip_root_deviation_l1",
        "base_height_l2",
        "undesired_contacts",
    )
    out = {}
    for name in names:
        rew = getattr(rewards, name, None)
        out[name] = None if rew is None else {"weight": value(rew, "weight"), "params": value(rew, "params")}
    return out


def main() -> None:
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

    from m2g_tooluse.assets.cfg.go2_piper import GO2PIPER_CFG, GO2PIPER_USD_PATH
    from m2g_tooluse.train.navigation.go2_piper_train_asset_cfg import (
        GO2PIPER_LIGHT_PIPER_CFG,
        GO2PIPER_TRAIN_CFG,
        GO2PIPER_TRAIN_USD_PATH,
    )
    from m2g_tooluse.train.navigation.go2_piper_velocity_env_cfg import Go2PiperVelocityFlatEnvCfg

    official_rough = UnitreeGo2RoughEnvCfg()
    official_flat = UnitreeGo2FlatEnvCfg()
    m2g_flat = Go2PiperVelocityFlatEnvCfg()

    dump(
        "Official UNITREE_GO2_CFG",
        {
            "usd_path": value(UNITREE_GO2_CFG, "spawn.usd_path"),
            "local_go2_usd_available": str(M2G_ROOT / "assets" / "robots" / "Go2" / "go2.usd"),
            "init_state.pos": value(UNITREE_GO2_CFG, "init_state.pos"),
            "joint_pos": value(UNITREE_GO2_CFG, "init_state.joint_pos"),
            "actuators": actuator_summary(UNITREE_GO2_CFG),
        },
    )

    dump(
        "Local GO2PIPER_CFG / GO2PIPER_TRAIN_CFG",
        {
            "GO2PIPER_USD_PATH": GO2PIPER_USD_PATH,
            "GO2PIPER_TRAIN_USD_PATH": GO2PIPER_TRAIN_USD_PATH,
            "GO2PIPER_CFG.init_state.pos": value(GO2PIPER_CFG, "init_state.pos"),
            "GO2PIPER_CFG.joint_pos": value(GO2PIPER_CFG, "init_state.joint_pos"),
            "GO2PIPER_CFG.actuators": actuator_summary(GO2PIPER_CFG),
            "GO2PIPER_TRAIN_CFG.init_state.pos": value(GO2PIPER_TRAIN_CFG, "init_state.pos"),
            "GO2PIPER_TRAIN_CFG.joint_pos": value(GO2PIPER_TRAIN_CFG, "init_state.joint_pos"),
            "GO2PIPER_TRAIN_CFG.actuators": actuator_summary(GO2PIPER_TRAIN_CFG),
            "GO2PIPER_LIGHT_PIPER_CFG.usd_path": value(GO2PIPER_LIGHT_PIPER_CFG, "spawn.usd_path"),
            "ArticulationRootAPI note": "Config comments say default prim is /go2_piper and ArticulationRootAPI is /go2_piper/go2/base; spawned Go2 base is {ENV_REGEX_NS}/Robot/go2/base.",
        },
    )

    def official_env_summary(task_id: str, cfg) -> dict:
        return {
            "task_id": task_id,
            "action_scale": value(cfg, "actions.joint_pos.scale"),
            "rewards": reward_summary(value(cfg, "rewards")),
            "reset_base.velocity_range": value(cfg, "events.reset_base.params.velocity_range"),
            "add_base_mass.range": value(cfg, "events.add_base_mass.params.mass_distribution_params"),
            "height_scanner.path": value(cfg, "scene.height_scanner.prim_path"),
            "base_contact.body_names": value(cfg, "terminations.base_contact.params.sensor_cfg.body_names"),
        }

    dump(
        "Official Go2 Env Cfgs",
        {
            "rough": official_env_summary("Isaac-Velocity-Rough-Unitree-Go2-v0", official_rough),
            "flat": official_env_summary("Isaac-Velocity-Flat-Unitree-Go2-v0", official_flat),
        },
    )

    dump(
        "Local Go2+Piper Velocity Env Cfg",
        {
            "task_id": "M2G-Navigation-Go2Piper-Velocity-Flat-v0",
            "action_joint_names": value(m2g_flat, "actions.joint_pos.joint_names"),
            "action_scale": value(m2g_flat, "actions.joint_pos.scale"),
            "command_ranges": {
                "lin_vel_x": value(m2g_flat, "commands.base_velocity.ranges.lin_vel_x"),
                "lin_vel_y": value(m2g_flat, "commands.base_velocity.ranges.lin_vel_y"),
                "ang_vel_z": value(m2g_flat, "commands.base_velocity.ranges.ang_vel_z"),
                "heading_command": value(m2g_flat, "commands.base_velocity.heading_command"),
            },
            "rewards": reward_summary(value(m2g_flat, "rewards")),
            "randomization": {
                "push_robot": value(m2g_flat, "events.push_robot"),
                "base_external_force_torque": value(m2g_flat, "events.base_external_force_torque"),
                "add_base_mass": value(m2g_flat, "events.add_base_mass"),
                "base_com": value(m2g_flat, "events.base_com"),
                "reset_robot_joints.params": value(m2g_flat, "events.reset_robot_joints.params"),
            },
            "terminations": {
                "base_contact.body_names": value(m2g_flat, "terminations.base_contact.params.sensor_cfg.body_names"),
                "bad_orientation": value(m2g_flat, "terminations.bad_orientation"),
                "base_link_height_below_minimum": value(m2g_flat, "terminations.base_link_height_below_minimum"),
            },
            "sensor_paths": {
                "height_scanner": value(m2g_flat, "scene.height_scanner.prim_path"),
                "contact_forces": value(m2g_flat, "scene.contact_forces.prim_path"),
            },
            "body_name_patches": {
                "base": "base",
                "feet": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                "thighs": ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"],
            },
        },
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
