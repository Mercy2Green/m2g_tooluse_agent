# ROSA And ROS2 Bridge

The ROSA process intentionally does not import IsaacLab. ROSA tools call ROS2 services, and the ROS2 bridge forwards `std_srvs/Trigger` calls to a local RPC server hosted inside the IsaacLab process.

Architecture:

```text
ROSA process
  -> ROS2 CLI / rclpy tools
  -> /m2g/gt_demo/* Trigger services
  -> m2g_gt_demo_bridge
  -> LocalGtDemoRpcServer
  -> IsaacLab controller
```

Services:

- `start`
- `reset`
- `go_to_object`
- `pick_object`
- `run_full_demo`
- `move_forward_policy`
- `stop_policy`
- `run_locomotion_test`
- `status`

Three-terminal flow:

Terminal A, IsaacLab + bridge:

```bash
cd $M2G_TOOLUSE_ROOT
PYTHONUNBUFFERED=1 TERM=xterm $ISAACLAB_PATH/isaaclab.sh -p \
  m2g_tooluse/gt_demo/run_gt_demo.py \
  --headless \
  --ros2-bridge
```

Terminal B, ROS2 CLI:

```bash
source /opt/ros/humble/setup.bash
ros2 service list | grep /m2g/gt_demo
ros2 service call /m2g/gt_demo/status std_srvs/srv/Trigger "{}"
```

Terminal C, ROSA agent:

```bash
source /opt/ros/humble/setup.bash
python m2g_tooluse/gt_demo/rosa_agent/m2g_rosa_agent.py --self-test-tools
```

Runner difference:

- `run_gt_demo.py` runs the GT pick demo.
- `run_locomotion_policy_demo.py` runs the RL locomotion policy test.

The bridge can expose service profiles, but callers must use commands supported by the active runner. TODO: service profile separation for gt_demo vs locomotion demo.

## Demo videos

Representative recordings (files under `videos/`):

- ROSA tools test — videos/test_rosa-2026-05-27_00.59.28.mp4
- ROS2 bridge test — videos/test_ros2-2026-05-27_00.52.14.mp4
