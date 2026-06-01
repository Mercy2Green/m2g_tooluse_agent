from __future__ import annotations

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import UniformVelocityCommandCfg

from m2g_tooluse.gt_demo.task.env_cfg import M2GGtDemoEnvCfg
from m2g_tooluse.train.navigation.go2_piper_velocity_env_cfg import GO2_LEG_JOINTS


@configclass
class LocomotionPolicyCommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.12, 0.45),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.35, 0.35),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class LocomotionPolicyObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_commands = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_LEG_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_LEG_JOINTS, preserve_order=True)},
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LocomotionPolicyActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=GO2_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class M2GGtDemoLocomotionPolicyEnvCfg(M2GGtDemoEnvCfg):
    """RL locomotion test in the same SimpleRoom scene as gt_demo.

    This task keeps the gt_demo room/object/container/Go2+Piper scene and reset
    behavior, but exposes the trained velocity-policy interface: 48-D policy
    observations, 12-D Go2 leg joint actions, and a runtime base_velocity
    command. The original SimpleRoom gt_demo task remains unchanged.
    """

    observations: LocomotionPolicyObservationsCfg = LocomotionPolicyObservationsCfg()
    actions: LocomotionPolicyActionsCfg = LocomotionPolicyActionsCfg()
    commands: LocomotionPolicyCommandsCfg = LocomotionPolicyCommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.0
