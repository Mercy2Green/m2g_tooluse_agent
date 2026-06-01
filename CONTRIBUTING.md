# Contributing

Please open issues with the environment, command, expected behavior, actual behavior, and relevant logs. Do not attach large assets, checkpoints, or private paths unless they are required and safe to share.

## Adding a Skill

Every new skill should include:

- A skill module
- A controller method
- A ROS2 service, or an explicit reason that a service is not needed
- A ROSA tool wrapper if it should be language-callable
- Documentation
- A smoke test

## Code Style

Keep Python simple and readable. Prefer existing project patterns over new abstractions. Use environment variables for local paths, especially `ISAACLAB_PATH`, `M2G_TOOLUSE_ROOT`, and `M2G_ASSET_ROOT`.

## Repository Hygiene

Do not commit large files, generated binaries, logs, videos, checkpoints, `.env`, API keys, or third-party assets. Verify asset licenses before documenting or sharing any derived files.

Run the relevant smoke tests before opening a PR:

```bash
python -m compileall m2g_tooluse
bash scripts/smoke_test_python.sh
python scripts/check_assets.py --asset-root assets
```
