from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Optional

from .types import VelocityLimits


ActionProvider = Callable[[Any], Any]


class IsaacLabVelocityCommandAdapter:
    """Adapter that writes base velocity commands into an IsaacLab command manager.

    IsaacLab exposes command tensors for reading through ``get_command``. Writing a
    teleop command requires touching the command term's backing tensor; that private
    access is intentionally isolated in this adapter so skills and scripts do not
    depend on command-manager internals.
    """

    def __init__(
        self,
        env: Any,
        command_name: str = "base_velocity",
        action_provider: Optional[ActionProvider] = None,
        allow_zero_action_step: bool = False,
    ):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.command_name = command_name
        self.action_provider = action_provider
        self.allow_zero_action_step = allow_zero_action_step
        self.dt = self._compute_dt()

    def set_velocity(
        self, vx: float, vy: float, wz: float, env_ids: Optional[Sequence[int]] = None
    ) -> None:
        """Set desired base velocity command for one or more IsaacLab envs."""
        command = self._command_tensor()
        if env_ids is None:
            command[:, 0] = vx
            command[:, 1] = vy
            command[:, 2] = wz
        else:
            command[list(env_ids), 0] = vx
            command[list(env_ids), 1] = vy
            command[list(env_ids), 2] = wz

    def stop(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Set zero base velocity command."""
        self.set_velocity(0.0, 0.0, 0.0, env_ids=env_ids)

    def step_for(self, duration_s: float) -> float:
        """Step the environment for a duration using a supplied action source.

        A velocity command only affects a locomotion policy if that policy consumes
        the command observation and produces low-level actions. When no policy or
        action provider is supplied, this method refuses to invent locomotion actions
        unless ``allow_zero_action_step`` was explicitly enabled for debug dry-runs.
        """
        duration_s = max(0.0, float(duration_s))
        if duration_s == 0.0:
            return 0.0
        steps = int(math.ceil(duration_s / self.dt))
        elapsed_s = 0.0
        obs = self._get_observations()
        for _ in range(steps):
            action = self._next_action(obs)
            obs = self._step(action)
            elapsed_s += self.dt
        return elapsed_s

    def velocity_limits(self) -> VelocityLimits:
        """Read velocity limits from the command cfg when available."""
        cfg = getattr(self._command_term(), "cfg", None)
        ranges = getattr(cfg, "limit_ranges", None) or getattr(cfg, "ranges", None)
        if ranges is None:
            return VelocityLimits()
        return VelocityLimits(
            vx=tuple(getattr(ranges, "lin_vel_x", VelocityLimits().vx)),
            vy=tuple(getattr(ranges, "lin_vel_y", VelocityLimits().vy)),
            wz=tuple(getattr(ranges, "ang_vel_z", VelocityLimits().wz)),
        )

    def _compute_dt(self) -> float:
        if hasattr(self.unwrapped, "step_dt"):
            return float(self.unwrapped.step_dt)
        cfg = getattr(self.unwrapped, "cfg", None)
        sim_cfg = getattr(cfg, "sim", None)
        if sim_cfg is not None and hasattr(sim_cfg, "dt") and hasattr(cfg, "decimation"):
            return float(sim_cfg.dt) * float(cfg.decimation)
        return 1.0 / 50.0

    def _command_tensor(self) -> Any:
        term = self._command_term()
        # Public read API returns the same tensor for UniformVelocityCommand in
        # current IsaacLab versions. If a future version returns a copy, fall back
        # to the term's backing tensor below.
        command_manager = getattr(self.unwrapped, "command_manager", None)
        if command_manager is not None and hasattr(command_manager, "get_command"):
            command = command_manager.get_command(self.command_name)
            if command is not None and hasattr(command, "__setitem__"):
                return command
        if hasattr(term, "vel_command_b"):
            return term.vel_command_b
        if hasattr(term, "command") and hasattr(term.command, "__setitem__"):
            return term.command
        raise RuntimeError(f"command '{self.command_name}' is not writable")

    def _command_term(self) -> Any:
        command_manager = getattr(self.unwrapped, "command_manager", None)
        if command_manager is None:
            raise RuntimeError("env has no command_manager; use the teleop env cfg or add a base_velocity command")
        if hasattr(command_manager, "get_term"):
            return command_manager.get_term(self.command_name)
        terms = getattr(command_manager, "_terms", None)
        if isinstance(terms, dict) and self.command_name in terms:
            return terms[self.command_name]
        raise RuntimeError(f"command_manager has no command term '{self.command_name}'")

    def _get_observations(self) -> Any:
        if hasattr(self.env, "get_observations"):
            return self.env.get_observations()
        if hasattr(self.unwrapped, "get_observations"):
            return self.unwrapped.get_observations()
        return None

    def _next_action(self, obs: Any) -> Any:
        if self.action_provider is not None:
            return self.action_provider(obs)
        if not self.allow_zero_action_step:
            raise RuntimeError("step_for requires an action_provider or allow_zero_action_step=True")
        action_space = getattr(self.env, "action_space", None)
        if action_space is None:
            raise RuntimeError("env has no action_space for debug zero-action stepping")
        try:
            import torch

            return torch.zeros(action_space.shape, device=self.unwrapped.device)
        except Exception as exc:
            raise RuntimeError(f"failed to create debug zero action: {exc}") from exc

    def _step(self, action: Any) -> Any:
        result = self.env.step(action)
        if isinstance(result, tuple) and result:
            return result[0]
        return result


def create_go2_nav_skill(env: Any, **adapter_kwargs: Any):
    """Create a reusable Go2 base navigation skill for an IsaacLab env."""
    from .go2_base_nav import Go2BaseNavSkill

    adapter = IsaacLabVelocityCommandAdapter(env, **adapter_kwargs)
    return Go2BaseNavSkill(adapter, limits=adapter.velocity_limits())


get_go2_base_nav_skill = create_go2_nav_skill
