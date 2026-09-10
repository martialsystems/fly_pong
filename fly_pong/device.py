"""FlyPongDevice: frozen encode, two gated readouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fly_pong.constants import load_constants
from fly_pong.features import AIM_KEYS, AIM_N, MOVE_KEYS, FeatureEncoder

from fly_pong.routers import SoftmaxRouter


def _move_bias() -> np.ndarray:
    b = np.zeros(len(MOVE_KEYS), dtype=np.float32)
    b[MOVE_KEYS.index("error_y")] = 2.5
    b[MOVE_KEYS.index("VS_down")] = 0.4
    b[MOVE_KEYS.index("VS_up")] = 0.4
    return b


def _aim_bias() -> np.ndarray:
    b = np.zeros(len(AIM_KEYS), dtype=np.float32)
    b[AIM_KEYS.index("desired_offset")] = 2.0
    b[AIM_KEYS.index("predicted_contact_y")] = 0.8
    b[AIM_KEYS.index("opponent_open_down")] = 0.5
    b[AIM_KEYS.index("opponent_open_up")] = 0.5
    return b


class FlyPongDevice:
    def __init__(self, aim_n: int = AIM_N, n_ommatidia: int = 32):
        self.encoder = FeatureEncoder(n_ommatidia)
        self.move_router = SoftmaxRouter(len(MOVE_KEYS), bias=_move_bias())
        self.aim_router = SoftmaxRouter(len(AIM_KEYS), bias=_aim_bias())
        self.aim_n = int(aim_n)
        self.C = load_constants()
        self._aim_setpoint = None

    def reset(self) -> None:
        self.encoder.reset()
        self._aim_setpoint = None

    def parameters_move(self):
        return self.move_router.parameters()

    def parameters_aim(self):
        return self.aim_router.parameters()

    def step_command(self, state: dict[str, Any]) -> dict[str, Any]:
        bank = self.encoder.encode(state)
        u_dy = self.move_router.command_np(bank.move)
        u_off = float(np.clip(self.aim_router.command_np(bank.aim), -1.0, 1.0))
        if abs(u_off) > 0.05:
            u_off = 1.0 if u_off > 0.0 else -1.0
        aim_active = bool(bank.incoming and bank.frames_to_paddle <= self.aim_n)
        speed = float(self.C["paddleSpeed"])
        ph = float(self.C["paddleH"])
        py = float(state.get("paddle_y", state.get("agent_y")))
        if not aim_active:
            self._aim_setpoint = None
        elif self._aim_setpoint is None:
            self._aim_setpoint = float(bank.predicted_contact_y_px)
        if aim_active:
            target_center = float(self._aim_setpoint) - u_off * (ph / 2.0)
            target_y = target_center - ph / 2.0
            err = target_y - py
            if err > 1.0:
                dy = speed
            elif err < -1.0:
                dy = -speed
            else:
                dy = 0.0
        else:
            if u_dy > 0.02:
                dy = speed
            elif u_dy < -0.02:
                dy = -speed
            else:
                dy = 0.0
        action = 0
        if dy < -0.5:
            action = 1
        elif dy > 0.5:
            action = 2
        return {
            "dy": dy,
            "action": action,
            "u_dy": u_dy,
            "u_offset": u_off,
            "aim_active": aim_active,
            "target_center_px": float(
                (self._aim_setpoint if self._aim_setpoint is not None else bank.predicted_contact_y_px)
                - u_off * (ph / 2.0)
            ),
            "paddle_center_px": float(py + ph / 2.0),
            "g_move": self.move_router.gates_np(),
            "g_aim": self.aim_router.gates_np(),
            "bank": bank,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "move": self.move_router.state_dict(),
                "aim": self.aim_router.state_dict(),
                "aim_n": self.aim_n,
            },
            path,
        )

    def load(self, path: Path, *, aim: bool = True) -> None:
        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.move_router.load_state_dict(blob["move"])
        if aim:
            try:
                self.aim_router.load_state_dict(blob["aim"])
            except RuntimeError:
                # Aim bank changed (clock channel removed). Keep move; reinit aim.
                pass
        self.aim_n = int(blob.get("aim_n", self.aim_n))
