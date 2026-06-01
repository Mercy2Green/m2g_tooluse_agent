from __future__ import annotations

"""ROS2 service bridge for the gt_demo local RPC server.

Run this with a ROS2 Humble-compatible Python, typically ``/usr/bin/python3``
after sourcing ``/opt/ros/humble/setup.bash``. It intentionally does not import
IsaacLab or ROSA.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from m2g_tooluse.gt_demo.ros2_bridge.rpc_server import send_rpc_command


GT_SERVICE_COMMANDS = {
    "/m2g/gt_demo/start": "start",
    "/m2g/gt_demo/reset": "reset",
    "/m2g/gt_demo/go_to_object": "go_to_object",
    "/m2g/gt_demo/pick_object": "pick_object",
    "/m2g/gt_demo/run_full_demo": "run_full_demo",
    "/m2g/gt_demo/status": "status",
}

LOCOMOTION_SERVICE_COMMANDS = {
    "/m2g/gt_demo/start": "start",
    "/m2g/gt_demo/reset": "reset",
    "/m2g/gt_demo/move_forward_policy": "move_forward_policy",
    "/m2g/gt_demo/stop_policy": "stop_policy",
    "/m2g/gt_demo/run_locomotion_test": "run_locomotion_test",
    "/m2g/gt_demo/status": "status",
}

SERVICE_PROFILES = {
    "gt": GT_SERVICE_COMMANDS,
    "locomotion": LOCOMOTION_SERVICE_COMMANDS,
    "all": {**GT_SERVICE_COMMANDS, **LOCOMOTION_SERVICE_COMMANDS},
}


class GtDemoBridgeNode(Node):
    def __init__(self, host: str, port: int, timeout_s: float, service_profile: str):
        super().__init__("m2g_gt_demo_bridge")
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.service_profile = service_profile
        service_commands = SERVICE_PROFILES[service_profile]
        self._services = []
        for service_name, command in service_commands.items():
            self._services.append(
                self.create_service(
                    Trigger,
                    service_name,
                    self._make_callback(command),
                )
            )
        self.get_logger().info(
            f"gt_demo bridge ready; rpc={self.host}:{self.port}; "
            f"profile={self.service_profile}; services={list(service_commands)}"
        )

    def _make_callback(self, command: str):
        def callback(_request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            try:
                result = send_rpc_command(command, host=self.host, port=self.port, timeout_s=self.timeout_s)
                response.success = bool(result.get("success", False))
                response.message = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as exc:
                response.success = False
                response.message = f"failed to call gt_demo rpc command '{command}': {exc}"
            return response

        return callback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 bridge for m2g gt_demo.")
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8765)
    parser.add_argument("--rpc-timeout", type=float, default=5.0)
    parser.add_argument(
        "--service-profile",
        choices=tuple(SERVICE_PROFILES),
        default="all",
        help="Which gt_demo service set to expose.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rclpy.init()
    node = GtDemoBridgeNode(args.rpc_host, args.rpc_port, args.rpc_timeout, args.service_profile)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
