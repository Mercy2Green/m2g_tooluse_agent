#!/usr/bin/env bash
set -euo pipefail

: "${ISAACLAB_PATH:?Set ISAACLAB_PATH=/path/to/IsaacLab}"
: "${M2G_GO2PIPER_POLICY_CKPT:?Set M2G_GO2PIPER_POLICY_CKPT=/path/to/checkpoint.pt}"
TERM="${TERM:-xterm}" "$ISAACLAB_PATH/isaaclab.sh" -p \
  m2g_tooluse/gt_demo/run_locomotion_policy_demo.py \
  --headless
