# Open Source Audit


## Source Repository State

`git status --short` did not print modified files during the initial scan.

## Large Files Found In Source

The source repository contains files larger than 50 MB, including:

- `assets/robots/Go2Piper/*.usd`
- third-party `.git/objects/pack/*.pack`
- third-party USD assets under `third_party/`

These were not copied.

## Asset, Checkpoint, And Video-Like Files Found In Source

The source repository contains many USD files under:

- `assets/piper_isaac_sim/`
- `assets/robots/Go2/`
- `assets/robots/Go2Piper/`
- `assets/room/`
- `third_party/`

The source repository also contains policy/checkpoint files under:

- `logs/rsl_rl/.../*.pt`
- `m2g_tooluse/gt_demo/ckpt/.../*.pt`
- `m2g_tooluse/train/navigation/checkpoints/.../*.pt`

These were not copied.

## Sensitive Content Found In Source

The source repository contains `apikey.md` with OpenAI-style keys. This file was not copied.

The keyword scan also matched API-token examples in third-party documentation and expected runtime environment reads in ROSA code. Only source code that reads API keys from environment variables was copied.


## Intentional Exclusions

- `assets/robots/`
- `assets/room/`
- `assets/piper_isaac_sim/`
- `assets/agx_arm_urdf/`
- `logs/`
- `outputs/`
- `results/`
- `videos/`
- `checkpoints/`
- `runs/`
- `wandb/`
- `apikey.md`
- `.env`
- `__pycache__/`
- `*.pt`, `*.pth`, `*.onnx`, `*.engine`
- `*.usd`, `*.usda`, `*.usdc`
- `*.mp4`, `*.mov`, `*.mkv`
- `*.db`, `*.log`

## Current Validation Notes

`python -m compileall m2g_tooluse` is used as the syntax check. Runtime IsaacLab imports still require an IsaacLab environment for actual demos and training.
