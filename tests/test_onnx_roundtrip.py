from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "artifacts" / "pong.onnx"
NORM = ROOT / "artifacts" / "norm.json"
MODEL = ROOT / "artifacts" / "pong_ppo.zip"


@pytest.mark.skipif(not ONNX.is_file() or not MODEL.is_file(), reason="train/export artifacts missing")
def test_onnx_matches_sb3_on_sample_obs():
    import onnxruntime as ort
    from stable_baselines3 import PPO

    from fly_pong.obs import apply_vecnorm

    norm = json.loads(NORM.read_text(encoding="utf-8"))
    model = PPO.load(str(MODEL), device="cpu")
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for _ in range(32):
        raw = rng.random(6).astype(np.float32)
        raw[2:4] = raw[2:4] * 2 - 1
        obs = apply_vecnorm(raw, norm)[None, :]
        sb3, _ = model.predict(obs, deterministic=True)
        onnx_out = sess.run(None, {"input": obs})
        onnx_action = int(np.asarray(onnx_out[0]).reshape(-1)[0])
        assert onnx_action == int(np.asarray(sb3).reshape(-1)[0])
