# Repository Design

Development rules:

- The `main` branch should stay runnable.
- Use feature branches for new skills.
- Third-party code stays in `third_party/` or is referenced externally.
- Assets and generated binaries stay out of Git.
- Do not mix IsaacLab process imports with ROSA process imports.

Every new skill should have:

- Skill module
- Controller method
- ROS2 service, or a clear reason not needed
- ROSA tool wrapper if language-callable
- Documentation
- Smoke test

The repository should keep source code reproducible while leaving large assets, checkpoints, logs, and local generated files outside Git.
