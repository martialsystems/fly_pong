# fly_pong

Can a fly-style vertical motion detector play Pong?

Motion alone cannot. T4/T5 on a 32-ommatidium strip, 20 games vs the env's lag paddle, seed 0: **0/20 matches, 0-220 points** (`logs/fly_gate_motion_only.json`). A still or sideways ball barely moves on a 1-D retina, so the paddle never acquires it.

Add a retinotopic centering reflex (luminance center of mass vs the paddle's height) and the same body won **40/40 matches, 439-46 points** at n=40, seed 0 (`logs/fly_gate.json`). One of those 40 hit the 20,000-frame cap at 10-4; the other 39 finished first to 11. Mean tracking error 0.023 of court height. That is a centering loop on a 32-pixel strip. The lock is vs the lag paddle (0.75× speed) built into `FlyPongEnv`.

## Results

Copied from the lock files under `logs/`.

| Controller | n | Matches | Points | Track err |
|------------|--:|--------:|-------:|----------:|
| T4/T5 motion only | 20 | 0/20 | 0-220 | 0.274 |
| T4/T5 + centering | 40 | 40/40 | 439-46 | 0.023 |

PPO on the same env, 100 games, was 99/100. That net is `public/models/pong.onnx`. It does not share weights with the fly loop.

## What this is

Gymnasium Pong, Discrete(3), observation `[ball_x, ball_y, ball_vx, ball_vy, agent_y, opp_y]`. Physics lives in `shared/constants.json`. Python and the browser clone the same step.

The fly loop: photoreceptor strip → L1/L2 half-wave → Reichardt T4c/d and T5c/d → a center-of-mass error against paddle height → stay / up / down. Approach gain is higher when `ball_vx` is toward the agent. `fly_pong/fbl_adapter.py` is a Neurokernel stub. Public FFBO servers do not execute circuits.

`public/` is a separate canvas opponent that loads the PPO ONNX in the browser.

## Run

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,train]"
.venv/bin/python scripts/sync_constants.py
.venv/bin/python -m pytest
```

Do not use stock `/usr/bin/python3 -m pytest`.

```bash
.venv/bin/python -m fly_pong.run_human
.venv/bin/python -m fly_pong.run_fly
.venv/bin/python scripts/eval_fly.py --games 40 --out logs/fly_gate.json
.venv/bin/python -m http.server 8000 --directory public
```

`eval_fly.py` rewrites the lock. Restamp the table from that JSON if the numbers move.

## Files

| Path | Role |
|------|------|
| `logs/fly_gate.json` | Centering lock, n=40 |
| `logs/fly_gate_motion_only.json` | Motion-only lock, n=20 |
| `fly_pong/physics.py` | Pure step, no pygame |
| `fly_pong/env.py` | Gymnasium wrapper |
| `fly_pong/sensors.py` | Vertical luminance strip |
| `fly_pong/brain.py` | T4/T5 Reichardt array |
| `fly_pong/bridge.py` | Centering + motion decode |
| `scripts/eval_fly.py` | Match win rate vs lag paddle |
| `scripts/train_pong.py` | PPO + VecNormalize |
| `public/` | Canvas PPO opponent |

Code is MIT.
