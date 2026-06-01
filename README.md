# M2G ToolUse: ROSA-Enabled Go2+Piper Tool-Use Simulation in IsaacLab

A research prototype for integrating IsaacLab, ROS2, ROSA, and skill-based control on a simulated Unitree Go2 + Piper mobile manipulator.

## What This Project Currently Supports

- Go2+Piper IsaacLab configuration
- GT search-and-pick demo
- ROS2 Trigger service bridge
- ROSA tool wrappers
- RL locomotion policy skill test
- Go2+Piper locomotion training configs
- Asset sanity and sanitization scripts

## What This Project Does Not Yet Provide

- Full SLAM
- Full Nav2 navigation
- Real perception
- Obstacle avoidance
- Robust sim-to-real locomotion
- Generalized grasping

## Repository Structure

```text
m2g_tooluse_agent/
  m2g_tooluse/        Python package with IsaacLab tasks, skills, demos, and training configs
  docs/               Public setup, demo, asset, training, and design documentation
  assets/             Asset manifest and placeholders only; no large third-party assets
  scripts/            Environment, asset, and smoke-test helpers
  examples/           ROSA prompts and ROS2 CLI examples
  third_party/        Notes for optional external dependencies; no vendored assets by default
```

## Requirements

- Ubuntu
- Isaac Sim / IsaacLab
- ROS2 Humble
- ROSA
- Python 3.10+

Use separate environments:

- `isaacsim5` for IsaacLab / IsaacSim
- `rosa` for ROSA and ROS2 CLI tools

IsaacLab, IsaacSim, ROS2, and ROSA are intentionally not declared as pip dependencies because they normally require external installation steps.

## Quick Start

```bash
git clone <your-new-repo-url> m2g_tooluse_agent
cd m2g_tooluse_agent

export ISAACLAB_PATH=/path/to/IsaacLab
export M2G_TOOLUSE_ROOT=$PWD
export M2G_ASSET_ROOT=$M2G_TOOLUSE_ROOT/assets
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

python -m compileall m2g_tooluse
bash scripts/smoke_test_python.sh
python scripts/check_assets.py --asset-root assets
```

Run the GT demo after assets are prepared:

```bash
TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/gt_demo/run_gt_demo.py \
  --headless \
  --object-name object
```

Run the locomotion policy demo after providing a checkpoint:

```bash
export M2G_GO2PIPER_POLICY_CKPT=/path/to/policy_checkpoint.pt
TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/gt_demo/run_locomotion_policy_demo.py \
  --headless
```

## Assets

Large robot, room, USD, checkpoint, log, and video files are not committed. See [docs/02_assets.md](docs/02_assets.md) for the expected layout, preparation flow, and license cautions.

## Demo Documentation

- [GT demo](docs/03_gt_demo.md)
- [RL locomotion policy](docs/04_rl_locomotion.md)
- [ROSA and ROS2 bridge](docs/05_rosa_ros2_bridge.md)
- [Training](docs/06_training.md)

## Demo videos (selected)

I added links to a small set of representative demo recordings in the docs. Selected clips (kept as examples in `videos/`):

- `videos/test_dk_run_demo-2026-05-28_01.26.43.mp4` — GT run demo
- `videos/test_RL-2026-05-27_02.14.14.mp4` — RL locomotion policy test
- `videos/test_ros2-2026-05-27_00.52.14.mp4` — ROS2 bridge test
- `videos/test_rosa-2026-05-27_00.59.28.mp4` — ROSA tools test

Open `videos/index.html` in a browser (see `videos/README.md`) to view these. Remove other files in `videos/` if you want to keep only the selected clips.

## Citation and License

See [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE).

The Apache-2.0 license applies to the source code in this repository. Third-party assets, robot descriptions, IsaacLab, IsaacSim, ROSA, Unitree, and AgileX/Piper assets are governed by their own licenses. This repository does not redistribute third-party robot or room assets by default.
