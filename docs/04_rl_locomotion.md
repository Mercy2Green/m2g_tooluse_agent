# RL Locomotion Policy

The locomotion runner is a policy skill test, not full navigation.

Task id:

```text
M2G-GT-Demo-Go2Piper-LocomotionPolicy-v0
```

Training cfg:

```text
Go2PiperVelocityFlatRosaSkillEnvCfg
```

Default command:

- `vx=0.25`
- `yaw_rate=0`
- `duration=3s`

The policy uses 12D Go2 leg actions. It does not teleport the root. Piper joints remain outside the locomotion policy action space.

Checkpoint variables:

```bash
export M2G_GO2PIPER_POLICY_CKPT=/path/to/checkpoint.pt
export M2G_GO2PIPER_POLICY_JIT=/path/to/policy_jit.pt
```

Single-machine test:

```bash
cd $M2G_TOOLUSE_ROOT
TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/gt_demo/run_locomotion_policy_demo.py \
  --headless
```

ROS2 service test:

```bash
TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/gt_demo/run_locomotion_policy_demo.py \
  --headless \
  --ros2-bridge

ros2 service call /m2g/gt_demo/run_locomotion_test std_srvs/srv/Trigger "{}"
```

ROSA prompt examples:

```text
Run the locomotion policy test and report whether the robot moved forward.
Move forward with the policy, stop, then return status.
```

Known limitations:

- Pure turn is not fully supported.
- No obstacle avoidance.
- Policy quality depends on the supplied checkpoint.
- No full SLAM/Nav2 integration yet.
