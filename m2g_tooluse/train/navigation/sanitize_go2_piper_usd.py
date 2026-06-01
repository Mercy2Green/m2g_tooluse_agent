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
    DEFAULT_SANITIZED_USD_PATH,
    DEFAULT_USD_PATH,
    M2G_TOOLUSE_ROOT,
    sanitize_usd,
    save_json_report,
)


def _resolve_output(path: str) -> Path:
    output = Path(path)
    if output.is_absolute():
        return output
    return M2G_TOOLUSE_ROOT / output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sanitized Go2+Piper USD for locomotion training.")
    parser.add_argument("--input", default=str(DEFAULT_USD_PATH))
    parser.add_argument("--output", default=str(DEFAULT_SANITIZED_USD_PATH))
    parser.add_argument("--scale-piper-mass", type=float, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = _resolve_output(args.output)
    report = sanitize_usd(input_path, output_path, scale_piper_mass=args.scale_piper_mass)
    report_path = DEFAULT_REPORT_DIR / (
        "sanitized_light_piper_patch_report.json"
        if args.scale_piper_mass is not None
        else "sanitized_patch_report.json"
    )
    save_json_report(report, report_path)

    print(f"input_path: {input_path}", flush=True)
    print(f"output_path: {output_path}", flush=True)
    print(f"scale_piper_mass: {args.scale_piper_mass}", flush=True)
    print(f"edit_count: {report['edit_count']}", flush=True)
    print(
        "fixed_non_finite_com_bodies: "
        f"{[edit['path'] for edit in report['edits'] if edit['field'] == 'centerOfMass']}",
        flush=True,
    )
    print(
        "fixed_zero_inertia_bodies: "
        f"{[edit['path'] for edit in report['edits'] if edit['field'] == 'diagonalInertia']}",
        flush=True,
    )
    print(f"patch_report_path: {report_path}", flush=True)


if __name__ == "__main__":
    main()
