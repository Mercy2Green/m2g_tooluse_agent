#!/usr/bin/env bash
set -euo pipefail

python -m compileall m2g_tooluse
python scripts/check_env.py
python scripts/check_assets.py --asset-root assets || true
