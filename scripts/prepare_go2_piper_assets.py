#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create local Go2+Piper asset folders and print preparation steps.")
    parser.add_argument("--asset-root", default="assets")
    args = parser.parse_args()
    root = Path(args.asset_root)
    for rel in ["robots/Go2", "robots/Piper", "robots/Go2Piper", "room", "cfg"]:
        (root / rel).mkdir(parents=True, exist_ok=True)
        print(f"created/checked {root / rel}")
    print("\nNext steps:")
    print("1. Place official Go2 USD under assets/robots/Go2/.")
    print("2. Place official Piper USD under assets/robots/Piper/.")
    print("3. Run sanity_check_go2_piper_usd.py and sanitize_go2_piper_usd.py with IsaacLab.")
    print("4. Run python scripts/check_assets.py --asset-root assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
