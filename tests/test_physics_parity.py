from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from fly_pong.obs import encode, encode_mirrored_right
from fly_pong.physics import dy_from_action, initial_state, lag_opponent_dy, step

ROOT = Path(__file__).resolve().parents[1]


def _build_spec(n: int = 240) -> dict:
    rng = np.random.default_rng(0)
    state = initial_state(serve_dir=1.0, angle=0.12)
    initial = json.loads(json.dumps(state))
    actions = []
    python_frames = []
    for _ in range(n):
        agent_action = int(rng.integers(0, 3))
        agent_dy = dy_from_action(agent_action)
        opp_dy = lag_opponent_dy(state)
        serve_angle = float(rng.uniform(-0.45, 0.45))
        actions.append(
            {
                "agent_action": agent_action,
                "agent_dy": agent_dy,
                "opp_dy": opp_dy,
                "serve_angle": serve_angle,
            }
        )
        state, reward = step(state, agent_dy, opp_dy, serve_angle=serve_angle)
        python_frames.append(
            {
                "state": json.loads(json.dumps(state)),
                "reward": reward,
                "obs": encode(state).tolist(),
                "obs_mirror": encode_mirrored_right(state).tolist(),
            }
        )
    return {"initial": initial, "actions": actions, "python_frames": python_frames}


def test_js_physics_matches_python(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        raise AssertionError("node is required for physics parity")
    spec = _build_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"initial": spec["initial"], "actions": spec["actions"]}))
    cli = ROOT / "public" / "js" / "parity_cli.mjs"
    proc = subprocess.run(
        [node, str(cli), str(spec_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT / "public" / "js"),
    )
    js = json.loads(proc.stdout)
    assert len(js["frames"]) == len(spec["python_frames"])
    keys = ("ball_x", "ball_y", "ball_vx", "ball_vy", "agent_y", "opp_y", "agent_score", "opp_score")
    for i, (py, js_frame) in enumerate(zip(spec["python_frames"], js["frames"])):
        for key in keys:
            assert js_frame["state"][key] == py["state"][key] or abs(
                js_frame["state"][key] - py["state"][key]
            ) < 1e-9, f"frame {i} {key}: py={py['state'][key]} js={js_frame['state'][key]}"
        assert abs(js_frame["reward"] - py["reward"]) < 1e-12
        np.testing.assert_allclose(js_frame["obs"], py["obs"], atol=1e-9)
        np.testing.assert_allclose(js_frame["obs_mirror"], py["obs_mirror"], atol=1e-9)
        assert bool(js_frame["state"]["terminated"]) == bool(py["state"]["terminated"])
