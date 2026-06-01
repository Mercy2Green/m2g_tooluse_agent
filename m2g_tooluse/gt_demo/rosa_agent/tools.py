from __future__ import annotations

import os
import subprocess
from typing import Sequence

from langchain_core.tools import StructuredTool


SERVICE_TYPE = "std_srvs/srv/Trigger"
GT_DEMO_SERVICES = {
    "start": "/m2g/gt_demo/start",
    "reset": "/m2g/gt_demo/reset",
    "go_to_object": "/m2g/gt_demo/go_to_object",
    "pick_object": "/m2g/gt_demo/pick_object",
    "run_full_demo": "/m2g/gt_demo/run_full_demo",
    "move_forward_policy": "/m2g/gt_demo/move_forward_policy",
    "stop_policy": "/m2g/gt_demo/stop_policy",
    "run_locomotion_test": "/m2g/gt_demo/run_locomotion_test",
    "status": "/m2g/gt_demo/status",
}


def _ros_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ROS_DOMAIN_ID", "0")
    env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    return env


def _run_ros2(args: Sequence[str], timeout_s: float = 20.0) -> str:
    cmd = ["ros2", *args]
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=_ros_env(),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        return (
            f"$ {' '.join(cmd)}\n"
            f"returncode=timeout\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
            f"timed out after {timeout_s} seconds"
        )
    output = (
        f"$ {' '.join(cmd)}\n"
        f"returncode={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return output


def list_gt_demo_services() -> str:
    """List ROS2 nodes and gt_demo services visible to the ROSA process."""
    nodes = _run_ros2(["node", "list"], timeout_s=10.0)
    services = _run_ros2(["service", "list"], timeout_s=10.0)
    gt_services = "\n".join(line for line in services.splitlines() if "/m2g/gt_demo" in line)
    return f"{nodes}\n\n{services}\n\ngt_demo services:\n{gt_services}"


def _call_trigger(service_name: str) -> str:
    return _run_ros2(["service", "call", service_name, SERVICE_TYPE, "{}"], timeout_s=20.0)


def start_gt_demo() -> str:
    """Queue/start the IsaacLab gt_demo controller."""
    return _call_trigger(GT_DEMO_SERVICES["start"])


def reset_gt_demo() -> str:
    """Reset the IsaacLab gt_demo environment."""
    return _call_trigger(GT_DEMO_SERVICES["reset"])


def move_go2_to_object() -> str:
    """Ask Go2 to move kinematically to the object standoff pose in gt_demo."""
    return _call_trigger(GT_DEMO_SERVICES["go_to_object"])


def pick_object_with_piper() -> str:
    """Ask Piper to run the scripted gt_demo pick step."""
    return _call_trigger(GT_DEMO_SERVICES["pick_object"])


def run_full_gt_demo() -> str:
    """Run the complete gt_demo: find object, move Go2, pick with Piper, verify."""
    return _call_trigger(GT_DEMO_SERVICES["run_full_demo"])


def move_forward_with_rl_policy() -> str:
    """Ask the trained Go2 leg RL locomotion policy to move forward with fixed test command."""
    return _call_trigger(GT_DEMO_SERVICES["move_forward_policy"])


def stop_rl_locomotion() -> str:
    """Ask the RL locomotion policy controller to command zero velocity briefly."""
    return _call_trigger(GT_DEMO_SERVICES["stop_policy"])


def run_rl_locomotion_test() -> str:
    """Run one fixed forward RL locomotion policy test and stop."""
    return _call_trigger(GT_DEMO_SERVICES["run_locomotion_test"])


def get_gt_demo_status() -> str:
    """Query the latest gt_demo controller status."""
    return _call_trigger(GT_DEMO_SERVICES["status"])


def create_gt_demo_rosa_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(list_gt_demo_services),
        StructuredTool.from_function(start_gt_demo),
        StructuredTool.from_function(reset_gt_demo),
        StructuredTool.from_function(move_go2_to_object),
        StructuredTool.from_function(pick_object_with_piper),
        StructuredTool.from_function(run_full_gt_demo),
        StructuredTool.from_function(move_forward_with_rl_policy),
        StructuredTool.from_function(stop_rl_locomotion),
        StructuredTool.from_function(run_rl_locomotion_test),
        StructuredTool.from_function(get_gt_demo_status),
    ]
