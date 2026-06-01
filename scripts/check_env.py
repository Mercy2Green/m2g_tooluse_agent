#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    names = ["ISAACLAB_PATH", "M2G_TOOLUSE_ROOT", "M2G_ASSET_ROOT", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION"]
    for name in names:
        value = os.environ.get(name)
        marker = "OK" if value else "MISSING"
        print(f"{marker:7} {name}={value or ''}")
    isaaclab = os.environ.get("ISAACLAB_PATH")
    if isaaclab and not (Path(isaaclab) / "isaaclab.sh").exists():
        print(f"WARNING ISAACLAB_PATH does not contain isaaclab.sh: {isaaclab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
