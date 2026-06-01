import gymnasium as gym

gym.register(
    id="m2g-go2-piper-simple-room-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_simple_room_env_cfg:Go2PiperSimpleRoomEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="M2G-ToolUse-Go2-SimpleRoom-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "m2g_tooluse.tasks.go2_simpleroom_teleop.go2_simpleroom_teleop_env_cfg:"
            "Go2SimpleRoomTeleopEnvCfg"
        ),
    },
    disable_env_checker=True,
)
