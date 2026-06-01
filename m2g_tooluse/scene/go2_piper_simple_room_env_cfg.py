from __future__ import annotations

import os
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from m2g_tooluse.assets.cfg.go2_piper import GO2PIPER_CFG

from humanoid.tasks.data.can import CAN_FANTA_CFG
from humanoid.tasks.data.container import CONTAINER_2X3_CFG

from humanoid.tasks.data.mug import MUG_CFG

from .simple_room_cfg import SimpleRoomSceneCfg


_DEFAULT_ASSET_ROOT = Path(os.environ.get("M2G_ASSET_ROOT", "assets"))
CAN_USD_PATH = str(_DEFAULT_ASSET_ROOT / "objects" / "can" / "can_fanta.usd")
CONTAINER_USD_PATH = str(_DEFAULT_ASSET_ROOT / "objects" / "container" / "container_2x3.usd")


@configclass
class ObjectTableSceneCfg(SimpleRoomSceneCfg):
    # Cans
    can_1: RigidObjectCfg = CAN_FANTA_CFG.replace(
        prim_path="/World/envs/env_.*/Can1",
    )
    # Initial position: Right side (y < 0)
    can_1.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.37, -0.3, 0.712),
        rot=(1.0, 0.0, 0.0, 0.0),
    )

    can_2: RigidObjectCfg = CAN_FANTA_CFG.replace(
        prim_path="/World/envs/env_.*/Can2",
    )
    # Initial position: Left side (y > 0)
    can_2.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.37, 0.3, 0.712),
        rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Container
    container: RigidObjectCfg = CONTAINER_2X3_CFG.replace(
        prim_path="/World/envs/env_.*/Container",
    )
    # Initial position: Center
    container.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.40, 0.0, 0.76),
        rot=(1.0, 0.0, 0.0, 0.0),
    )

    robot: ArticulationCfg = GO2PIPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-2.0, 0.0, -0.7),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
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
                "joint7": 0.05,
                "joint8": -0.05,
            },
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=base_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        mug_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("can_1")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

@configclass
class Go2PiperSimpleRoomEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2
