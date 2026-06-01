# Overview

M2G ToolUse is a research prototype for a simulated Unitree Go2 + Piper mobile manipulator in IsaacLab. The repository keeps source code, adapters, configs, and reproducible preparation scripts in Git while leaving large or third-party assets outside the repository.

Core components:

- IsaacLab scene and task configs for Go2+Piper
- Ground-truth staged search-and-pick demo
- ROS2 `std_srvs/Trigger` bridge
- ROSA tool wrappers
- RL locomotion policy test runner
- Go2+Piper locomotion training configs and RSL-RL wrappers
- USD sanity and sanitization utilities

This first public skeleton favors clarity and extensibility over one-command reproduction.
