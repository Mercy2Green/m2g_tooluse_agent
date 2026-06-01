# Installation

Use separate environments so IsaacLab dependencies do not conflict with ROSA and ROS2 CLI tooling.

```bash
conda create -n codex_isaacsim5 python=3.10
conda create -n codex_rosa python=3.10
```

Install Isaac Sim and IsaacLab following their official documentation. Install ROS2 Humble and source it before using bridge commands:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Configure paths:

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export M2G_TOOLUSE_ROOT=/path/to/m2g_tooluse_agent
export M2G_ASSET_ROOT=$M2G_TOOLUSE_ROOT/assets
```

Run basic checks:

```bash
cd $M2G_TOOLUSE_ROOT
python -m compileall m2g_tooluse
bash scripts/smoke_test_python.sh
```
