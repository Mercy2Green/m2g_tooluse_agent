from __future__ import annotations

import gymnasium as gym

from m2g_tooluse.gt_demo.config import GT_DEMO_TASK_ID


gym.register(
    id=GT_DEMO_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "m2g_tooluse.gt_demo.task.env_cfg:M2GGtDemoEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="M2G-GT-Demo-Go2Piper-LocomotionPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "m2g_tooluse.gt_demo.task.locomotion_env_cfg:M2GGtDemoLocomotionPolicyEnvCfg",
        "rsl_rl_cfg_entry_point": "m2g_tooluse.train.navigation.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
    disable_env_checker=True,
)
