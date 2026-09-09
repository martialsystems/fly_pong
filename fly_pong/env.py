"""Gymnasium Pong. Observation is the 6-D vector; info['state'] feeds the fly bridge."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fly_pong.constants import load_constants
from fly_pong.obs import OBS_HIGH, OBS_LOW, encode
from fly_pong.physics import (
    dy_from_action,
    initial_state,
    lag_opponent_dy,
    step as physics_step,
)


class FlyPongEnv(gym.Env):
    """One-player Pong vs a lagging chase opponent. Gymnasium API."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        max_score: int | None = None,
    ):
        super().__init__()
        self.C = load_constants()
        self.width = int(self.C["width"])
        self.height = int(self.C["height"])
        self.max_score = int(self.C["maxScore"] if max_score is None else max_score)
        self.C = dict(self.C)
        self.C["maxScore"] = self.max_score
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)

        self._state: dict[str, Any] = initial_state(self.C, serve_dir=1.0, angle=0.0)
        self._screen = None
        self._clock = None
        self._pygame = None

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

    def get_state_dict(self) -> dict[str, Any]:
        s = self._state
        vy = float(s["ball_vy"])
        if vy > 0:
            direction = "down"
        elif vy < 0:
            direction = "up"
        else:
            direction = "still"
        return {
            "ball_x": float(s["ball_x"]),
            "ball_y": float(s["ball_y"]),
            "ball_vx": float(s["ball_vx"]),
            "ball_vy": float(s["ball_vy"]),
            "paddle_y": float(s["agent_y"]),
            "opp_y": float(s["opp_y"]),
            "ball_direction": direction,
            "width": self.width,
            "height": self.height,
            "paddle_h": int(self.C["paddleH"]),
            "agent_score": int(s["agent_score"]),
            "opp_score": int(s["opp_score"]),
        }

    def _obs(self) -> np.ndarray:
        return encode(self._state, self.C)

    def _info(self) -> dict[str, Any]:
        return {"state": self.get_state_dict()}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        direction = float(self.np_random.choice(np.array([-1.0, 1.0])))
        angle = float(self.np_random.uniform(-float(self.C["serveAngleMax"]), float(self.C["serveAngleMax"])))
        self._state = initial_state(self.C, serve_dir=direction, angle=angle)
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), self._info()

    def step(self, action):
        agent_dy = dy_from_action(int(action), C=self.C)
        opp_dy = lag_opponent_dy(self._state, self.C)
        angle = float(
            self.np_random.uniform(-float(self.C["serveAngleMax"]), float(self.C["serveAngleMax"]))
        )
        prev_vx = float(self._state["ball_vx"])
        self._state, reward = physics_step(
            self._state, agent_dy, opp_dy, self.C, serve_angle=angle
        )
        ph = float(self.C["paddleH"])
        center = float(self._state["agent_y"]) + ph / 2.0
        track_err = abs(center - float(self._state["ball_y"])) / float(self.height)
        reward -= 0.05 * track_err
        if prev_vx < 0 and float(self._state["ball_vx"]) > 0:
            reward += 0.5
        terminated = bool(self._state["terminated"])
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), float(reward), terminated, False, self._info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "human":
            self._render_frame()
            return None
        return None

    def _ensure_pygame(self):
        if self._pygame is not None:
            return self._pygame
        import pygame

        self._pygame = pygame
        return pygame

    def _render_frame(self):
        pygame = self._ensure_pygame()
        C = self.C
        if self._screen is None:
            pygame.init()
            pygame.display.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Fly Pong")
            else:
                self._screen = pygame.Surface((self.width, self.height))
            self._clock = pygame.time.Clock()

        self._screen.fill((12, 12, 18))
        s = self._state
        pygame.draw.rect(
            self._screen,
            (230, 230, 230),
            (int(C["agentX"]), int(s["agent_y"]), int(C["paddleW"]), int(C["paddleH"])),
        )
        pygame.draw.rect(
            self._screen,
            (180, 180, 200),
            (int(C["oppX"]), int(s["opp_y"]), int(C["paddleW"]), int(C["paddleH"])),
        )
        pygame.draw.circle(
            self._screen,
            (255, 210, 80),
            (int(s["ball_x"]), int(s["ball_y"])),
            int(C["ballR"]),
        )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(self._screen)), (1, 0, 2)).copy()

    def close(self):
        if self._screen is not None and self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
            self._screen = None
            self._clock = None
            self._pygame = None
