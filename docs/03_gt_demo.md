# GT Demo

The GT demo is a staged ground-truth demo, not a complete autonomy stack. It does not provide SLAM, Nav2, real perception, obstacle avoidance, or generalized grasping.

Pipeline:

```text
find_object_gt
navigate_to_grasp_standoff_gt
plan_fixed_grasp_gt
execute_piper_pick_gt
verify_pick_gt
```

`go_to_object` currently defaults to kinematic debug base movement. It is useful for isolating the manipulation and bridge stack, not for proving physical navigation.

Task id:

```text
M2G-GT-Demo-Go2Piper-SimpleRoom-v0
```

Single-machine headless run:

```bash
cd $M2G_TOOLUSE_ROOT
TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/gt_demo/run_gt_demo.py \
  --headless \
  --object-name object
```

IsaacLab + ROS2 bridge:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd $M2G_TOOLUSE_ROOT
PYTHONUNBUFFERED=1 TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/gt_demo/run_gt_demo.py \
  --headless \
  --object-name object \
  --ros2-bridge
```

ROSA tools without LLM:

```bash
source /opt/ros/humble/setup.bash
cd $M2G_TOOLUSE_ROOT
python m2g_tooluse/gt_demo/rosa_agent/m2g_rosa_agent.py --self-test-tools
python m2g_tooluse/gt_demo/rosa_agent/m2g_rosa_agent.py --self-test-run-full-demo
```

Troubleshooting:

- Missing assets: run `python scripts/check_assets.py --asset-root assets`.
- ROS2 services not visible: check `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, and that Terminal A sourced ROS2.
- `TERM=dumb`: run commands with `TERM=xterm`.
- ROSA natural-language mode requires `OPENAI_API_KEY` or a compatible API configuration.

## Demo videos

Key recorded demos (see `videos/`):

- GT run demo — videos/test_dk_run_demo-2026-05-28_01.26.43.mp4
- ROS2 bridge demo — videos/test_ros2-2026-05-27_00.52.14.mp4

You can open `videos/index.html` for a simple gallery. I selected the GT and ROS2 bridge clips as the most representative; remove other files in `videos/` if you want to keep only these.
