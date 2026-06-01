# Copyright (c) 2022-2026.
# SPDX-License-Identifier: BSD-3-Clause
#
# Isaac Lab articulation config for the locally merged Unitree Go2 + AgileX Piper USD.
#
# This config assumes the USD default prim is:
#   /go2_piper
# and the actual ArticulationRootAPI is on:
#   /go2_piper/go2/base
#
# Typical use in an Isaac Lab scene config:
#
#   from .go2_piper import GO2PIPER_CFG
#   self.scene.robot = GO2PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#
# After spawning at "{ENV_REGEX_NS}/Robot", the Go2 base link will be:
#   {ENV_REGEX_NS}/Robot/go2/base

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_DEFAULT_ASSET_ROOT = Path(os.environ.get("M2G_ASSET_ROOT", "assets"))
GO2PIPER_USD_PATH = str(_DEFAULT_ASSET_ROOT / "robots" / "Go2Piper" / "go2_piper_v1.usd")


GO2PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2PIPER_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=16,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Same nominal base height as the IsaacLab Unitree Go2 config.
        # Adjust this if your mounted Piper changes the initial clearance in your scene.
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            # -----------------------------
            # Unitree Go2 legs
            # -----------------------------
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,

            # -----------------------------
            # AgileX Piper arm
            # Your USD uses joint1..joint6 directly.
            # -----------------------------
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,

            # -----------------------------
            # AgileX Piper gripper
            # Your USD uses prismatic joint7/joint8, not piper_finger_joint7/8.
            # Open gripper: joint7 at +0.05, joint8 at -0.05.
            # Closed gripper would usually be joint7=0.0, joint8=0.0.
            # -----------------------------
            "joint7": 0.05,
            "joint8": -0.05,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # -----------------------------
        # Unitree Go2 actuators
        #
        # Split hip/thigh and calf because the merged USD exposes different drive
        # maxForce and maxJointVelocity values for calf joints.
        # # -----------------------------
        ### The origional
        "go2_hip_thigh": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
            ],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "go2_calf": DCMotorCfg(
            joint_names_expr=[
                ".*_calf_joint",
            ],
            effort_limit=45.43,
            saturation_effort=45.43,
            velocity_limit=15.7,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),

        # #### The harder one
        # "go2_hip": DCMotorCfg(
        #     joint_names_expr=[
        #         ".*_hip_joint",
        #     ],
        #     effort_limit=45.0,
        #     saturation_effort=45.0,
        #     velocity_limit=35.0,
        #     stiffness=60.0,
        #     damping=2.0,
        #     friction=0.0,
        # ),

        # "go2_thigh": DCMotorCfg(
        #     joint_names_expr=[
        #         ".*_thigh_joint",
        #     ],
        #     effort_limit=70.0,
        #     saturation_effort=70.0,
        #     velocity_limit=40.0,
        #     stiffness=90.0,
        #     damping=3.5,
        #     friction=0.0,
        # ),

        # "go2_calf": DCMotorCfg(
        #     joint_names_expr=[
        #         ".*_calf_joint",
        #     ],
        #     effort_limit=90.0,
        #     saturation_effort=90.0,
        #     velocity_limit=40.0,
        #     stiffness=110.0,
        #     damping=4.5,
        #     friction=0.0,
        # ),

        # "go2_hip": DCMotorCfg(
        #     joint_names_expr=[
        #         ".*_hip_joint",
        #     ],
        #     effort_limit=500.0,
        #     saturation_effort=500.0,
        #     velocity_limit=35.0,
        #     stiffness=60.0,
        #     damping=2.0,
        #     friction=0.0,
        # ),

        # "go2_thigh": DCMotorCfg(
        #     joint_names_expr=[
        #         ".*_thigh_joint",
        #     ],
        #     effort_limit=500.0,
        #     saturation_effort=500.0,
        #     velocity_limit=40.0,
        #     stiffness=90.0,
        #     damping=3.5,
        #     friction=0.0,
        # ),

        # "go2_calf": DCMotorCfg(
        #     joint_names_expr=[
        #         ".*_calf_joint",
        #     ],
        #     effort_limit=500.0,
        #     saturation_effort=500.0,
        #     velocity_limit=40.0,
        #     stiffness=110.0,
        #     damping=4.5,
        #     friction=0.0,
        # ),

        # -----------------------------
        # AgileX Piper arm actuators
        #
        # Values are adapted from the available Piper IsaacLab-style config,
        # but joint names are corrected to match your merged USD.
        # -----------------------------
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint[1-6]",
            ],
            effort_limit=25.0,
            velocity_limit=1.5,
            stiffness={
                "joint1": 200.0,
                "joint2": 170.0,
                "joint3": 120.0,
                "joint4": 80.0,
                "joint5": 50.0,
                "joint6": 20.0,
            },
            damping={
                "joint1": 100.0,
                "joint2": 60.0,
                "joint3": 70.0,
                "joint4": 24.0,
                "joint5": 20.0,
                "joint6": 10.0,
            },
        ),
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint7",
                "joint8",
            ],
            effort_limit_sim=22.0,
            velocity_limit_sim=1.5,
            stiffness=800.0,
            damping=20.0,
        ),
    },
)
