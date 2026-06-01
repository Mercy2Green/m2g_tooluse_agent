from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


@configclass
class Go2PiperFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """RSL-RL PPO runner cfg for the Go2+Piper flat velocity task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.experiment_name = "m2g_go2_piper_velocity_flat"
        self.run_name = ""
        self.max_iterations = 1000
        self.save_interval = 100
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


@configclass
class Go2PiperFlatOfficialGo2WarmstartPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """Runner cfg that matches the official IsaacLab Go2 flat checkpoint network shape."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.experiment_name = "m2g_go2_piper_velocity_flat"
        self.run_name = ""
        self.max_iterations = 1000
        self.save_interval = 100
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
