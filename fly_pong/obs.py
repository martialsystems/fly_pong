"""6-D observation used by PPO training and the browser opponent."""

from __future__ import annotations

from typing import Any

import numpy as np

from fly_pong.constants import load_constants
from fly_pong.physics import clip, paddle_span


OBS_LOW = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
OBS_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
OBS_KEYS = ("ball_x", "ball_y", "ball_vx", "ball_vy", "agent_y", "opp_y")


def encode(state: dict[str, Any], C: dict[str, Any] | None = None) -> np.ndarray:
    """Training observation: agent is the left paddle."""
    C = C or load_constants()
    span = paddle_span(C)
    scale = float(C["velScale"])
    return np.array(
        [
            clip(float(state["ball_x"]) / float(C["width"]), 0.0, 1.0),
            clip(float(state["ball_y"]) / float(C["height"]), 0.0, 1.0),
            clip(float(state["ball_vx"]) / scale, -1.0, 1.0),
            clip(float(state["ball_vy"]) / scale, -1.0, 1.0),
            clip(float(state["agent_y"]) / span, 0.0, 1.0),
            clip(float(state["opp_y"]) / span, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def encode_mirrored_right(state: dict[str, Any], C: dict[str, Any] | None = None) -> np.ndarray:
    """Observation for a left-trained policy that now plays the right paddle.

    Flip x and vx, then treat the right paddle as the policy's agent.
    """
    C = C or load_constants()
    span = paddle_span(C)
    scale = float(C["velScale"])
    return np.array(
        [
            clip(1.0 - float(state["ball_x"]) / float(C["width"]), 0.0, 1.0),
            clip(float(state["ball_y"]) / float(C["height"]), 0.0, 1.0),
            clip((-float(state["ball_vx"])) / scale, -1.0, 1.0),
            clip(float(state["ball_vy"]) / scale, -1.0, 1.0),
            clip(float(state["opp_y"]) / span, 0.0, 1.0),
            clip(float(state["agent_y"]) / span, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def apply_vecnorm(obs: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(norm["mean"], dtype=np.float32)
    var = np.asarray(norm["var"], dtype=np.float32)
    eps = float(norm.get("epsilon", 1e-8))
    clip_obs = float(norm.get("clip_obs", 10.0))
    std = np.sqrt(var + eps)
    scaled = (np.asarray(obs, dtype=np.float32) - mean) / std
    return np.clip(scaled, -clip_obs, clip_obs).astype(np.float32)
