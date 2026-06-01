from __future__ import annotations

"""Local RSL-RL train wrapper for the Go2+Piper velocity task."""

import runpy
import sys
import os
from pathlib import Path


DEFAULT_TASK = "M2G-Navigation-Go2Piper-Velocity-Flat-Fixed-v0"
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "IsaacLab"))
OFFICIAL_RSL_RL_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
OFFICIAL_TRAIN = OFFICIAL_RSL_RL_DIR / "train.py"


def _ensure_default_task() -> None:
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", DEFAULT_TASK])


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[3]
    for path in (workspace_root, OFFICIAL_RSL_RL_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    _ensure_default_task()

    # Register custom Gym task before the official script applies hydra_task_config.
    import m2g_tooluse.train.navigation  # noqa: F401

    runpy.run_path(str(OFFICIAL_TRAIN), run_name="__main__")


if __name__ == "__main__":
    main()
