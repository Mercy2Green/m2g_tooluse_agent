# ROS2 CLI Examples

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 node list
ros2 service list | grep /m2g/gt_demo
ros2 service call /m2g/gt_demo/status std_srvs/srv/Trigger "{}"
ros2 service call /m2g/gt_demo/run_full_demo std_srvs/srv/Trigger "{}"
ros2 service call /m2g/gt_demo/run_locomotion_test std_srvs/srv/Trigger "{}"
```
