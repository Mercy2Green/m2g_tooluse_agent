# Troubleshooting

- IsaacLab import failed: use the IsaacLab environment and launch scripts through `$ISAACLAB_PATH/isaaclab.sh`.
- ROS2 services not visible: source `/opt/ros/humble/setup.bash` in every ROS2 terminal.
- `ROS_DOMAIN_ID` mismatch: set the same value in IsaacLab, ROS2 CLI, and ROSA terminals.
- RMW implementation mismatch: set `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` consistently unless your system requires another implementation.
- Policy checkpoint missing: set `M2G_GO2PIPER_POLICY_CKPT` or `M2G_GO2PIPER_POLICY_JIT`.
- Action/obs dim mismatch: inspect the selected task and checkpoint; Go2+Piper locomotion expects 12D Go2 leg actions.
- Assets missing: run `python scripts/check_assets.py --asset-root assets` and follow `docs/02_assets.md`.
- USD fails to load: verify file paths under `M2G_ASSET_ROOT`, upstream licenses, and sanitized USD generation.
- Robot collapses or policy stays still: check asset articulation root, mass/inertia values, action joint set, and checkpoint curriculum.
- Terminal type `dumb`: run with `TERM=xterm`.
- No OpenAI key for ROSA natural-language mode: set `OPENAI_API_KEY` or use the no-LLM self-test commands.
