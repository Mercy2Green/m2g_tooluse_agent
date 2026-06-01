from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from m2g_tooluse.scene.go2_piper_simple_room_env_cfg import (
    ActionsCfg,
    EventCfg,
    ObjectTableSceneCfg,
    ObservationsCfg as BaseObservationsCfg,
    TerminationsCfg,
)


@configclass
class CommandsCfg:
    """Base velocity command for Go2 teleop in the simple room."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Teleop observations include the generated base velocity command."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2SimpleRoomTeleopEnvCfg(ManagerBasedRLEnvCfg):
    """Simple-room Go2 teleop task with a writable base velocity command."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    rewards = None
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2
