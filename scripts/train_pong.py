#!/usr/bin/env python3
"""Train PPO on FlyPongEnv. Offline only; the site loads the exported ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fly_pong.env import FlyPongEnv

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def make_env():
    return FlyPongEnv(render_mode=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([make_env] * 8)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=int(args.timesteps))
    model.save(str(ARTIFACTS / "pong_ppo"))
    env.save(str(ARTIFACTS / "pong_vecnormalize.pkl"))
    print(f"saved {ARTIFACTS / 'pong_ppo.zip'} and vecnormalize")


if __name__ == "__main__":
    main()
