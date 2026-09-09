# Fly Pong

A Gymnasium Pong environment, a small PPO policy exported to ONNX, and a static page where a human plays that policy in the browser.

Observation is `[ball_x, ball_y, ball_vx, ball_vy, agent_y, opp_y]`, positions in `[0, 1]`, velocities in `[-1, 1]`. Actions are Discrete(3): stay, up, down. Physics constants live in `shared/constants.json`. Python and JavaScript step the same formulas.

The public page is `public/index.html`. Inference is onnxruntime-web plus a skill slider (reaction delay and random no-ops). If the ONNX file or WASM runtime fails to load, the right paddle uses the lag-chase heuristic. There is no server-side inference.

A Python lab loop (`python -m fly_pong.run_fly`) maps the ball to a vertical luminance strip and a T4/T5-style Reichardt detector. FlyBrainLab is a stub in `fly_pong/fbl_adapter.py` until a local Neurokernel session exists.

## How to run

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,train]"
.venv/bin/python scripts/sync_constants.py
.venv/bin/python -m pytest
```

Do not use stock `/usr/bin/python3 -m pytest`.

Human physics check (opens a pygame window):

```bash
.venv/bin/python -m fly_pong.run_human
```

Fly-circuit closed loop:

```bash
.venv/bin/python -m fly_pong.run_fly
```

Static site (any static server):

```bash
.venv/bin/python -m http.server 8000 --directory public
```

Then open `http://127.0.0.1:8000/`. Arrow keys or W/S, or drag on the court.

## Train and export

```bash
.venv/bin/python scripts/train_pong.py --timesteps 300000
.venv/bin/python scripts/eval_pong.py --games 100
.venv/bin/python scripts/export_onnx.py
```

Eval prints win rate vs the lag opponent. Export writes `public/models/pong.onnx` and `public/models/norm.json`. The browser AI plays the right paddle, so it mirrors x and vx before the net (the policy was trained as the left agent).

## Files

| Path | Role |
|------|------|
| `shared/constants.json` | Width, speeds, bounce, observation scale |
| `fly_pong/physics.py` | Pure step, no pygame |
| `fly_pong/env.py` | Gymnasium wrapper |
| `fly_pong/obs.py` | 6-D vector and right-paddle mirror |
| `fly_pong/bridge.py` | Lab T4/T5 interface |
| `scripts/train_pong.py` | PPO + VecNormalize |
| `scripts/export_onnx.py` | ONNX + `norm.json` |
| `public/` | Canvas opponent, no Python |
| `scripts/viewport_sanity.py` | Phone 390×844 and desktop ~1280 via CDP |

Original code is MIT.
