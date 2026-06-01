from __future__ import annotations

import gymnasium as gym


def _register(
    id_: str,
    cfg_name: str,
    rsl_rl_cfg_name: str = "Go2PiperFlatPPORunnerCfg",
) -> None:
    gym.register(
        id=id_,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "m2g_tooluse.train.navigation.go2_piper_velocity_env_cfg:" f"{cfg_name}",
            "rsl_rl_cfg_entry_point": "m2g_tooluse.train.navigation.rsl_rl_ppo_cfg:" f"{rsl_rl_cfg_name}",
        },
    )


_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-v0",
    "Go2PiperVelocityFlatEnvCfg",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-StandStill-v0",
    "Go2PiperVelocityFlatStandStillEnvCfg",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-Upright-v0",
    "Go2PiperVelocityFlatForwardUprightEnvCfg",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-RosaTurn-v0",
    "Go2PiperVelocityFlatRosaTurnEnvCfg",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-RosaSkill-v0",
    "Go2PiperVelocityFlatRosaSkillEnvCfg",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-Play-v0",
    "Go2PiperVelocityFlatEnvCfg_PLAY",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-Fixed-v0",
    "Go2PiperVelocityFlatEnvCfg",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-Fixed-Play-v0",
    "Go2PiperVelocityFlatEnvCfg_PLAY",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-NoBaseContactDebug-v0",
    "Go2PiperVelocityFlatNoBaseContactDebugEnvCfg",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-LightPiperDebug-v0",
    "Go2PiperVelocityFlatLightPiperDebugEnvCfg",
)

_register(
    "M2G-Navigation-Go2Piper-Velocity-Flat-OfficialGo2Warmstart-v0",
    "Go2PiperVelocityFlatEnvCfg",
    "Go2PiperFlatOfficialGo2WarmstartPPORunnerCfg",
)


gym.register(
    id="M2G-Navigation-Go2Piper-Velocity-Flat-Legacy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "m2g_tooluse.train.navigation.go2_piper_velocity_env_cfg:"
        "Go2PiperVelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "m2g_tooluse.train.navigation.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="M2G-Navigation-Go2Piper-Velocity-Flat-Legacy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "m2g_tooluse.train.navigation.go2_piper_velocity_env_cfg:"
        "Go2PiperVelocityFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "m2g_tooluse.train.navigation.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
)
