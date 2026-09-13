#!/usr/bin/env python3
"""Export PPO policy + VecNormalize stats to artifacts/. Offline only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fly_pong.env import FlyPongEnv
from fly_pong.obs import apply_vecnorm

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self.policy(observation, deterministic=True)


def make_env():
    return FlyPongEnv(render_mode=None)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(str(ARTIFACTS / "pong_vecnormalize.pkl"), env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(ARTIFACTS / "pong_ppo"), env=env, device="cpu")

    rms = env.obs_rms
    norm = {
        "mean": rms.mean.astype(float).tolist(),
        "var": rms.var.astype(float).tolist(),
        "epsilon": float(env.epsilon),
        "clip_obs": float(env.clip_obs),
    }
    (ARTIFACTS / "norm.json").write_text(json.dumps(norm, indent=2) + "\n", encoding="utf-8")

    onnx_path = ARTIFACTS / "pong.onnx"
    dummy = th.randn(1, 6)
    th.onnx.export(
        OnnxableSB3Policy(model.policy),
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["actions", "values", "log_prob"],
        dynamo=False,  # torch 2.9+ defaults to dynamo and needs onnxscript
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    raw = np.zeros(6, dtype=np.float32)
    normalized = apply_vecnorm(raw, norm)[None, :]
    sb3_action, _ = model.predict(normalized, deterministic=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": normalized.astype(np.float32)})
    onnx_action = int(np.asarray(onnx_out[0]).reshape(-1)[0])
    sb3_int = int(np.asarray(sb3_action).reshape(-1)[0])
    if onnx_action != sb3_int:
        raise SystemExit(f"ONNX action {onnx_action} != SB3 {sb3_int}")
    print(f"exported {onnx_path} and {ARTIFACTS / 'norm.json'}; sample action={onnx_action}")


if __name__ == "__main__":
    main()
