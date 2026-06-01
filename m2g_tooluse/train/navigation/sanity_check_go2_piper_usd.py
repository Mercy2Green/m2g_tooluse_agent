from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _go2_piper_usd_tools import (
    DEFAULT_REPORT_DIR,
    DEFAULT_USD_PATH,
    build_usd_report,
    save_json_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Go2+Piper USD physics metadata.")
    parser.add_argument("--usd", default=str(DEFAULT_USD_PATH))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    report = build_usd_report(args.usd)
    report_path = Path(args.report_dir) / "go2_piper_usd_sanity_report.json"
    save_json_report(report, report_path)

    print(f"usd_path: {report['usd_path']}", flush=True)
    print(f"articulation_roots: {report['articulation_roots']}", flush=True)
    print(f"rigid_body_count: {report['rigid_body_count']}", flush=True)
    print(f"joint_count: {report['joint_count']}", flush=True)
    print(f"actuated_joints: {report['actuated_joints']}", flush=True)
    print(f"body_names_containing_base: {report['body_names_containing_base']}", flush=True)
    print(f"body_names_containing_foot: {report['body_names_containing_foot']}", flush=True)
    print(f"body_names_containing_thigh: {report['body_names_containing_thigh']}", flush=True)
    print(f"body_names_containing_calflower: {report['body_names_containing_calflower']}", flush=True)
    print(
        "non_finite_centerOfMass_bodies: "
        f"{[body['body_name'] for body in report['non_finite_centerOfMass_bodies']]}",
        flush=True,
    )
    print(
        "exact_zero_diagonalInertia_bodies: "
        f"{[body['body_name'] for body in report['exact_zero_diagonalInertia_bodies']]}",
        flush=True,
    )
    print(f"piper_total_mass: {report['piper_total_mass']}", flush=True)
    print(f"go2_total_mass: {report['go2_total_mass']}", flush=True)
    print(f"total_robot_mass: {report['total_robot_mass']}", flush=True)
    print(
        "fixed_joint_from_go2_base_to_piper_arm_base: "
        f"{report['fixed_joint_from_go2_base_to_piper_arm_base']}",
        flush=True,
    )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}", flush=True)
    print(f"report_path: {report_path}", flush=True)


if __name__ == "__main__":
    main()
