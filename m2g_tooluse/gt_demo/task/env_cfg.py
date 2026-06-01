from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from m2g_tooluse.gt_demo.scene.scene_cfg import GtDemoObjectTableSceneCfg


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})

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
class M2GGtDemoEnvCfg(ManagerBasedRLEnvCfg):
    scene: GtDemoObjectTableSceneCfg = GtDemoObjectTableSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2
