# Training

The Go2+Piper locomotion training package registers flat velocity tasks for the merged articulation. The policy controls only the 12 Go2 leg joints; Piper joints remain at nominal/default poses.

Task ids:

- `M2G-Navigation-Go2Piper-Velocity-Flat-v0`
- `M2G-Navigation-Go2Piper-Velocity-Flat-StandStill-v0`
- `M2G-Navigation-Go2Piper-Velocity-Flat-Upright-v0`
- `M2G-Navigation-Go2Piper-Velocity-Flat-RosaTurn-v0`
- `M2G-Navigation-Go2Piper-Velocity-Flat-RosaSkill-v0`
- `M2G-Navigation-Go2Piper-Velocity-Flat-Play-v0`

Training env configs include `Go2PiperVelocityFlatRosaSkillEnvCfg`. PPO settings live in `m2g_tooluse/train/navigation/rsl_rl_ppo_cfg.py`.

Official Go2 checkpoint warm-start: use a locally obtained IsaacLab Go2 checkpoint as initialization if its license allows your use. Do not commit it.

Train:

```bash
cd $M2G_TOOLUSE_ROOT
TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/train/navigation/train.py \
  --task M2G-Navigation-Go2Piper-Velocity-Flat-RosaSkill-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 4000
```

Resume:

```bash
TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/train/navigation/train.py \
  --task M2G-Navigation-Go2Piper-Velocity-Flat-RosaSkill-v0 \
  --resume \
  --load_run <run_name> \
  --checkpoint <checkpoint_name.pt>
```

Play/export:

```bash
TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/train/navigation/play.py \
  --task M2G-Navigation-Go2Piper-Velocity-Flat-Play-v0 \
  --checkpoint /path/to/model.pt \
  --headless \
  --num_envs 1
```

Logs are written under `logs/rsl_rl/` and ignored by Git.

Reward tuning notes:

- Start with standstill/upright stages before skill-style commands.
- Watch base height, thigh contacts, command tracking, and action smoothness.
- Validate that Piper joints do not enter the action space.

Limitations:

- Policy quality depends on asset quality and checkpoint curriculum.
- Pure turning and stop/reposition behavior are still experimental.
- This is not a full navigation stack.
