#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_ASSETS = {
    "go2_usd": "assets/robots/Go2/go2.usd",
    "piper_usd": "assets/robots/Piper/piper_v2.usd",
    "go2_piper_sanitized_usd": "assets/robots/Go2Piper/go2_piper_v1_train_sanitized.usd",
    "room_usd": "assets/room/simple_room.usd",
}


def _read_manifest_paths(manifest: Path) -> dict[str, str]:
    try:
        import yaml
    except Exception:
        print("[check_assets] PyYAML is not installed; using built-in default asset list.")
        return DEFAULT_ASSETS

    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    assets = data.get("assets", {})
    paths = {}
    for name, spec in assets.items():
        if isinstance(spec, dict) and spec.get("required", True):
            paths[name] = str(spec["path"])
    return paths or DEFAULT_ASSETS


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local M2G ToolUse assets without downloading anything.")
    parser.add_argument("--asset-root", default="assets", help="Asset root directory.")
    parser.add_argument("--manifest", default="assets/manifest.example.yaml", help="Asset manifest path.")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).resolve()
    manifest = Path(args.manifest)
    paths = _read_manifest_paths(manifest) if manifest.exists() else DEFAULT_ASSETS

    missing = []
    for name, rel_path in paths.items():
        rel = Path(rel_path)
        candidate = asset_root / rel.relative_to("assets") if rel.parts and rel.parts[0] == "assets" else asset_root / rel
        if candidate.exists():
            print(f"[check_assets] OK      {name}: {candidate}")
        else:
            print(f"[check_assets] MISSING {name}: {candidate}")
            missing.append((name, candidate))

    if missing:
        print("\nRequired assets are missing. This repository does not download or redistribute third-party assets.")
        print("Follow docs/02_assets.md to obtain official assets, generate merged Go2+Piper USDs, and rerun this check.")
        return 1

    print("[check_assets] all required assets are present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
