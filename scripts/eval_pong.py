#!/usr/bin/env python3
"""Win rate of a saved PPO policy vs the lag opponent."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fly_pong.env import FlyPongEnv
from fly_pong.obs import apply_vecnorm

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _norm_from_vec(path: Path) -> dict:
    venv = DummyVecEnv([lambda: FlyPongEnv(render_mode=None)])
    vec = VecNormalize.load(str(path), venv)
    return {
        "mean": np.asarray(vec.obs_rms.mean, dtype=np.float32),
        "var": np.asarray(vec.obs_rms.var, dtype=np.float32),
        "epsilon": float(vec.epsilon),
        "clip_obs": float(vec.clip_obs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--model", type=Path, default=ARTIFACTS / "pong_ppo")
    parser.add_argument("--vecnorm", type=Path, default=ARTIFACTS / "pong_vecnormalize.pkl")
    args = parser.parse_args()

    model = PPO.load(str(args.model), device="cpu")
    norm = _norm_from_vec(args.vecnorm)

    wins = 0
    losses = 0
    for i in range(int(args.games)):
        env = FlyPongEnv(render_mode=None)
        obs, _ = env.reset(seed=i)
        while True:
            obs_n = apply_vecnorm(obs, norm)
            action, _ = model.predict(obs_n, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        state = env.unwrapped._state
        if int(state["agent_score"]) > int(state["opp_score"]):
            wins += 1
        else:
            losses += 1
        env.close()
    rate = wins / float(args.games)
    print(f"wins={wins} losses={losses} win_rate={rate:.3f} n={args.games}")
    if rate < 0.9:
        raise SystemExit(f"win rate {rate:.3f} below 0.9; train longer")


if __name__ == "__main__":
    main()
