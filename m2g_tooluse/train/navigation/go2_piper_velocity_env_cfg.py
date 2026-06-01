from __future__ import annotations

from isaaclab.envs.mdp.rewards import undesired_contacts
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from m2g_tooluse.train.navigation.go2_piper_train_asset_cfg import (
    GO2PIPER_LIGHT_PIPER_CFG,
    GO2PIPER_TRAIN_CFG,
)
from m2g_tooluse.train.navigation import mdp as nav_mdp

from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp


GO2_LEG_JOINTS = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

GO2_BASE_BODY = "base"

GO2_FOOT_BODIES = [
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
]

GO2_THIGH_BODIES = [
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
]

# Subset for hip/root joints (external rotation typically appears here)
GO2_HIP_JOINTS = GO2_LEG_JOINTS[:4]


@configclass
class Go2PiperVelocityFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Official Go2 flat velocity cfg with the local Go2+Piper articulation.

    The policy action space is deliberately restricted to the 12 Go2 leg joints.
    Piper joints remain in the articulation and are reset to their configured
    nominal pose, but they are not part of the policy output.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = GO2PIPER_TRAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos.joint_names = GO2_LEG_JOINTS
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True

        if getattr(self.scene, "height_scanner", None) is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/go2/base"
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/go2/.*"

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.50)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # lin_vel_x = (-0.4, 0.4)
        # lin_vel_y = (-0.15, 0.15)
        # ang_vel_z = (-0.4, 0.4)

        self._patch_action_observation_reward_filters()
        self._patch_body_name_patterns()
        self._reduce_randomization()


        ##### The adding reward
        # add imports at top of file:
        # from isaaclab.managers import RewardTermCfg as RewTerm
        # import isaaclab.envs.mdp as base_mdp
        # import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

        # 调参顺序建议（稳一点）：

        # 先只加 base_height_l2 + 增强 flat_orientation_l2，跑 100 iter 看是否还趴。
        # 再加 stand_still，防止低速时塌。
        # 最后开 undesired_contacts，抑制大腿触地捷径。

        # # 1) stronger upright bias
        # self.rewards.flat_orientation_l2.weight = -5.0

        # 2) keep base around standing height
        self.rewards.base_height_l2 = RewTerm(
            func=base_mdp.base_height_l2,
            weight=-0.1,
            params={
                "target_height": 0.28,   # first guess; tune by log
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # 2.1) keep hip-root joints unconstrained during warmup so the policy can learn to walk.
        self.rewards.hip_root_deviation_l1 = None

        # 2.2) make velocity tracking sharper for the narrower command range.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25

        # 3) terminate when base orientation is too far from upright
        # limit_angle in radians; ~1.0 rad (~57 deg) is a reasonable starting point
        self.terminations.bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 1.0})

        # 3.1) terminate when the selected base link height drops too low
        self.terminations.base_link_height_below_minimum = DoneTerm(
            func=nav_mdp.body_height_below_minimum,
            params={
                "minimum_height": 0.05,
                "asset_cfg": SceneEntityCfg("robot", body_names=GO2_BASE_BODY),
            },
        )


        # # 3) discourage crouched/collapsed idle posture (only when command is small)
        # self.rewards.stand_still = RewTerm(
        #     func=loco_mdp.stand_still_joint_deviation_l1,
        #     weight=-0.25,
        #     params={
        #         "command_name": "base_velocity",
        #         "command_threshold": 0.10,
        #         "asset_cfg": self._go2_joint_cfg(),
        #     },
        # )

        # # 4) punish thigh contacts to avoid prone solution
        # self.rewards.undesired_contacts = RewTerm(
        #     func=base_mdp.undesired_contacts,
        #     weight=-0.6,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2_THIGH_BODIES),
        #         "threshold": 1.0,
        #     },
        # )


    def _go2_joint_cfg(self) -> SceneEntityCfg:
        return SceneEntityCfg("robot", joint_names=GO2_LEG_JOINTS, preserve_order=True)

    def _patch_action_observation_reward_filters(self) -> None:
        """Restrict policy joint observations and joint penalties to Go2 leg joints."""
        self.observations.policy.joint_pos.params = {"asset_cfg": self._go2_joint_cfg()}
        self.observations.policy.joint_vel.params = {"asset_cfg": self._go2_joint_cfg()}

        for reward_name in ("dof_torques_l2", "dof_acc_l2", "dof_pos_limits"):
            reward = getattr(self.rewards, reward_name, None)
            if reward is not None:
                reward.params = {"asset_cfg": self._go2_joint_cfg()}

    def _patch_body_name_patterns(self) -> None:
        """Patch body regexes for the nested Go2 base in the merged articulation."""
        if getattr(self.events, "add_base_mass", None) is not None:
            self.events.add_base_mass.params["asset_cfg"].body_names = GO2_BASE_BODY
        if getattr(self.events, "base_com", None) is not None:
            self.events.base_com.params["asset_cfg"].body_names = GO2_BASE_BODY
        if getattr(self.events, "base_external_force_torque", None) is not None:
            self.events.base_external_force_torque.params["asset_cfg"].body_names = GO2_BASE_BODY
        if getattr(self.terminations, "base_contact", None) is not None:
            self.terminations.base_contact.params["sensor_cfg"].body_names = GO2_BASE_BODY
        if getattr(self.rewards, "feet_air_time", None) is not None:
            self.rewards.feet_air_time.params["sensor_cfg"].body_names = GO2_FOOT_BODIES
        if getattr(self.rewards, "undesired_contacts", None) is not None:
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = GO2_THIGH_BODIES

    def _reduce_randomization(self) -> None:
        """Keep the first fixed task focused on asset/contact correctness."""
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class Go2PiperVelocityFlatStandStillEnvCfg(Go2PiperVelocityFlatEnvCfg):
    """Stage 0: stand upright in place without relying on crawling."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Stage 0 should be a pure standing task, initialized from the official Go2 checkpoint.
        # The command space is zeroed so the policy learns to hold posture in place.
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Keep the robot upright and at a standing height.
        # target_height: 0.26-0.30 is the useful sweep; start at 0.28.
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.base_height_l2 = None
        self.rewards.base_body_height_l2 = RewTerm(
            func=nav_mdp.body_height_l2,
            weight=-1.0,
            params={
                "target_height": 0.28,
                "asset_cfg": SceneEntityCfg("robot", body_names=GO2_BASE_BODY),
            },
        )
        # self.rewards.base_body_height_exp = RewTerm(
        #     func=nav_mdp.body_height_exp,
        #     weight=2,
        #     params={
        #         "target_height": 0.28,
        #         "std": 0.06,
        #         "asset_cfg": SceneEntityCfg("robot", body_names=GO2_BASE_BODY),
        #     },
        # )

        # Standing should not depend on hip-root compensation.
        self.rewards.hip_root_deviation_l1 = None

        # Zero-velocity commands should reward stillness, not gait.
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Prevent prone shuffling from being considered a valid standing strategy.
        # minimum_height: 0.16-0.20 is the practical sweep; start at 0.18.
        self.terminations.base_link_height_below_minimum.params["minimum_height"] = 0.2

        # Add a direct stillness reward and a light thigh-contact penalty.
        # command_threshold: 0.04-0.08 is the useful sweep; start at 0.05.
        self.rewards.stand_still = RewTerm(
            func=loco_mdp.stand_still_joint_deviation_l1,
            weight=-0.25,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "asset_cfg": self._go2_joint_cfg(),
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-0.3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2_THIGH_BODIES),
                "threshold": 1.0,
            },
        )


@configclass
class Go2PiperVelocityFlatForwardUprightEnvCfg(Go2PiperVelocityFlatEnvCfg):
    """Stage 1: upright forward walking without prone crawling."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep the command set forward-only; this stage should recover upright gait first.
        # lin_vel_x: use a narrow positive band so the policy cannot hide behind standing.
        # lin_vel_y: keep at 0.0 until the robot can walk upright.
        # ang_vel_z: keep at 0.0 until forward gait is stable.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Tighten heading stability first; if this is still too loose, try -3.0 to -4.0.
        self.rewards.flat_orientation_l2.weight = -2.5

        # Replace the root-height penalty with a body-based version.
        # target_height: start around 0.26-0.30; lower values encourage crouch, higher values can reset too often.
        self.rewards.base_height_l2 = None
        # self.rewards.base_body_height_exp = RewTerm(
        #     func=nav_mdp.body_height_exp,
        #     weight=2,
        #     params={
        #         "target_height": 0.28,
        #         "std": 0.06,
        #         "asset_cfg": SceneEntityCfg("robot", body_names=GO2_BASE_BODY),
        #     },
        # )
        # Keep the policy focused on standing-up locomotion, not hip-root compensation.
        self.rewards.base_body_height_l2 = RewTerm(
            func=nav_mdp.body_height_l2,
            weight=-1.0,
            params={
                "target_height": 0.28,
                "asset_cfg": SceneEntityCfg("robot", body_names=GO2_BASE_BODY),
            },
        )
        self.rewards.hip_root_deviation_l1 = None

        # Track forward velocity sharply, but keep the band narrow enough that the robot does not dive.
        # std: 0.20-0.30 is a reasonable sweep; start at 0.25.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25

        # yaw tracking stays on but weak in this stage; if forward gait is stable, 0.5-1.0 is the usual sweep.
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25

        # Do not allow a body-link to skim the floor and count as "walking".
        # minimum_height: 0.16-0.20 is the practical sweep; start at 0.16.
        self.terminations.base_link_height_below_minimum.params["minimum_height"] = 0.2

        # Penalize thigh contact lightly to remove the prone/crawling shortcut.
        # weight: -0.2 is the first pass; -0.4 and -0.6 are the next escalation points.
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-0.2,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2_THIGH_BODIES),
                "threshold": 1.0,
            },
        )


@configclass
class Go2PiperVelocityFlatRosaTurnEnvCfg(Go2PiperVelocityFlatForwardUprightEnvCfg):
    """Stage 2: low-speed forward walking with yaw tracking for ROSA turns."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep lateral velocity off; ROSA can use turn + forward for pose reaching.
        # lin_vel_x: allow slow starts and easy stopping.
        # ang_vel_z: modest yaw range so in-place turns are learnable.
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.40)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.35, 0.35)

        # Yaw becomes a first-class objective here; 1.0-1.5 is the usual sweep once turns are stable.
        self.rewards.track_ang_vel_z_exp.weight = 1.2
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25

        # Relax forward tracking slightly because yaw + forward is harder than straight walking.
        self.rewards.track_lin_vel_xy_exp.weight = 1.7
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.30


@configclass
class Go2PiperVelocityFlatRosaSkillEnvCfg(Go2PiperVelocityFlatRosaTurnEnvCfg):
    """Stage 3a: ROSA move + turn, but avoid the stationary local optimum."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Avoid teaching the policy that standing still is the dominant behavior.
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Must move forward; no reverse/lateral yet.
        self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.35, 0.35)

        # Do not over-amplify near-zero command reward.
        self.rewards.track_lin_vel_xy_exp.weight = 2.2
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25

        self.rewards.track_ang_vel_z_exp.weight = 1.2
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25

        # Do not train stop in the same phase.
        self.rewards.stand_still = None

        # Keep anti-crawling constraints.
        self.rewards.base_body_height_l2.weight = -1.5
        self.terminations.base_link_height_below_minimum.params["minimum_height"] = 0.10

# @configclass
# class Go2PiperVelocityFlatRosaSkillEnvCfg(Go2PiperVelocityFlatRosaTurnEnvCfg):
#     """Stage 3: low-speed base skill policy for ROSA move/turn/stop commands."""

#     def __post_init__(self) -> None:
#         super().__post_init__()

#         # Add a little standing and a tiny lateral band only after upright walking is reliable.
#         # lin_vel_x: include a small reverse band for stop/reposition skills.
#         # lin_vel_y: keep this tiny; do not jump straight to full omnidirectional walking.
#         # ang_vel_z: widen slightly for ROSA turn commands.
#         self.commands.base_velocity.rel_standing_envs = 0.10
#         self.commands.base_velocity.ranges.lin_vel_x = (-0.10, 0.45)
#         self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
#         self.commands.base_velocity.ranges.ang_vel_z = (-0.45, 0.45)


#         # Track forward velocity sharply, but keep the band narrow enough that the robot does not dive.
#         # std: 0.20-0.30 is a reasonable sweep; start at 0.25.
#         self.rewards.track_lin_vel_xy_exp.weight = 4.0
#         self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25

#         # yaw tracking stays on but weak in this stage; if forward gait is stable, 0.5-1.0 is the usual sweep.
#         self.rewards.track_ang_vel_z_exp.weight = 4
#         self.rewards.track_ang_vel_z_exp.params["std"] = 0.25



#         # Stop/stand behavior helps the policy finish a command cleanly instead of lingering in motion.
#         # command_threshold: 0.06-0.10 is the practical sweep; start at 0.08.
#         self.rewards.stand_still = RewTerm(
#             func=loco_mdp.stand_still_joint_deviation_l1,
#             weight=-0.10,
#             params={
#                 "command_name": "base_velocity",
#                 "command_threshold": 0.08,
#                 "asset_cfg": self._go2_joint_cfg(),
#             },
#         )

#         # If crawling starts to reappear, tighten the height penalty instead of changing the command space first.
#         # weight: -1.0 to -1.5 is the next sweep; keep -1.5 as the upper end before you add more contact pressure.
#         self.rewards.base_body_height_l2.weight = -1.5

#         # Slightly tighten the minimum height once the forward gait is already upright.
#         self.terminations.base_link_height_below_minimum.params["minimum_height"] = 0.18


@configclass
class Go2PiperVelocityFlatEnvCfg_PLAY(Go2PiperVelocityFlatEnvCfg):
    """Play/debug variant for Go2+Piper flat velocity locomotion."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2PiperVelocityFlatNoBaseContactDebugEnvCfg(Go2PiperVelocityFlatEnvCfg):
    """Diagnostic task with base-contact termination disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.base_contact = None


@configclass
class Go2PiperVelocityFlatLightPiperDebugEnvCfg(Go2PiperVelocityFlatEnvCfg):
    """Diagnostic task using the optional mass-scaled Piper sanitized USD."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = GO2PIPER_LIGHT_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
