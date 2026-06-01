#!/usr/bin/env bash
set -euo pipefail

: "${ISAACLAB_PATH:?Set ISAACLAB_PATH=/path/to/IsaacLab}"
TERM="${TERM:-xterm}" "$ISAACLAB_PATH/isaaclab.sh" -p \
  m2g_tooluse/gt_demo/run_gt_demo.py \
  --headless \
  --object-name object
