# Assets

## Why Assets Are Not Committed

Robot, room, and USD assets may be large, generated, or governed by third-party licenses. Users must obtain them from official sources and reproduce generated or sanitized assets locally. This repository does not redistribute Unitree, AgileX/Piper, IsaacLab, IsaacSim, room, checkpoint, video, or log assets by default.

## Required Layout

```text
assets/
  robots/
    Go2/
    Piper/
    Go2Piper/
  room/
  cfg/
```

## Go2 Asset

If using the IsaacLab official Unitree Go2 asset, it may come from the IsaacLab/Nucleus asset path. Do not redistribute it unless its license permits redistribution.

## Piper Asset

Obtain Piper/AgileX assets from the official upstream source. Keep original files local under `assets/robots/Piper/`; do not commit them to Git.

## Go2+Piper Merged Asset Generation

Intended local process:

1. Place the original Go2 USD under `assets/robots/Go2/`.
2. Place the original Piper USD under `assets/robots/Piper/`.
3. Generate or update the Go2+Piper wrapper/merged USD under `assets/robots/Go2Piper/`.
4. Run `m2g_tooluse/train/navigation/sanity_check_go2_piper_usd.py`.
5. Run `m2g_tooluse/train/navigation/sanitize_go2_piper_usd.py`.
6. Write the sanitized training USD, for example `assets/robots/Go2Piper/go2_piper_v1_train_sanitized.usd`.
7. Run `m2g_tooluse/train/navigation/inspect_go2_piper_locomotion_env.py`.

Current scripts:

- `m2g_tooluse/train/navigation/sanity_check_go2_piper_usd.py`
- `m2g_tooluse/train/navigation/sanitize_go2_piper_usd.py`
- `m2g_tooluse/train/navigation/inspect_go2_piper_locomotion_env.py`

The public code now reads common asset paths through `M2G_ASSET_ROOT`, but some helper scripts may still need CLI polish for every asset variant. TODO: add a complete wrapper merge script once the upstream asset licenses and canonical filenames are finalized.

Check local assets:

```bash
python scripts/check_assets.py --asset-root assets
```
