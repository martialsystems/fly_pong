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

## Device (move then aim)

Named channels, softmax gates, T4/T5 frozen. MoveRouter runs every frame. AimRouter only in the last 10 frames before contact. Bounce is still the geometric paddle offset in `physics.py`.

Copied from `logs/device_move.json` and `logs/device_aim.json`, n=40, seed 0, lag paddle:

| Phase | Matches | Points | Contact | Open-hit | Biggest gate |
|-------|--------:|-------:|--------:|---------:|--------------|
| A move | 40/40 | 422-48 | 0.992 | 0.520 | error_y 0.984 |
| B aim vs lag | 40/40 | 420-47 | 0.992 | 0.497 | predicted_contact_t 0.900 |

Phase A mastered tracking via error_y. Phase B vs lag did not beat chase at placement (open-hit ~0.5). The clock channel `predicted_contact_t` is out of the aim bank.

Vs a frozen Phase A returner, same 40 seeds (`logs/aim_hypothesis.json` then `logs/aim_vs_returner.json`): move-only 12/40, 264-289, open-hit 0.531. Move+aim 18/40, 316-321, open-hit 0.568. Point delta +52 cleared a loose OR-bar. Open-hit +0.037 did not. That is not placement.

Signed-open eval (`logs/aim_sign_hypothesis.json` written first, then `logs/aim_sign.json`): sign(geo) vs open side 0.469 → 0.542 (Δ +0.073, bar +0.10). Open-hit Δ still +0.037 (bar +0.08). |geo| 0.78 with signed-open 0.54 is a random corner-smasher: fat offsets, half of them still to the covered side. Landing distance vs the far paddle fell 163 → 140. Command sign matched the gap 0.65 of the time; the bounce did not. Pass failed. Self-play remains locked.

Aim is coupled to bounce (|geo| 0.78) and buys points vs a tracker (+52) without clearing placement (open-hit +0.037). Self-play remains locked.

PPO on the 6-number box, 100 games, was 99/100 (`public/models/pong.onnx`). That net does not share weights with the fly loop or the device.

## What this is

Gymnasium Pong. Physics in `shared/constants.json`. Python and the browser clone the same step.

Hand loop: photoreceptor strip → L1/L2 half-wave → Reichardt T4c/d and T5c/d → centering. Device: the same strip plus `error_y`, time-to-paddle, LC-like blob energy, opponent open space, `desired_offset`. `pongforge/` refuses aim-before-move and refuses lag-as-finished-title claims.

`fly_pong/fbl_adapter.py` is a Neurokernel stub. Public FFBO servers do not execute circuits.

`public/` is a canvas opponent that loads the PPO ONNX.

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
PYTHONPATH=. .venv/bin/python scripts/train_move.py
PYTHONPATH=. .venv/bin/python scripts/eval_device.py --out logs/device_move.json
PYTHONPATH=. .venv/bin/python scripts/train_aim.py
PYTHONPATH=. .venv/bin/python scripts/eval_aim_returner.py
PYTHONPATH=. .venv/bin/python scripts/eval_aim_sign.py
.venv/bin/python -m http.server 8000 --directory public
```

Restamp tables from the JSON if the numbers move.

## Files

| Path | Role |
|------|------|
| `logs/fly_gate.json` | Centering lock, n=40 |
| `logs/fly_gate_motion_only.json` | Motion-only lock, n=20 |
| `logs/device_move.json` | Phase A lock |
| `logs/device_aim.json` | Phase B vs lag (null placement) |
| `logs/aim_hypothesis.json` | Pass/fail written before the returner eval |
| `logs/aim_vs_returner.json` | Move-only vs move+aim vs Phase A (thin point pass) |
| `logs/aim_sign_hypothesis.json` | Signed-open pass/fail, written first |
| `logs/aim_sign.json` | Signed-open vs Phase A (fail; corner-smasher) |
| `fly_pong/features.py` | Named move/aim channels |
| `fly_pong/routers.py` | Softmax gates |
| `fly_pong/device.py` | Encode + two heads |
| `pongforge/` | Phase order and claim bans |
| `scripts/eval_fly.py` | Hand-loop match rate |
| `scripts/eval_device.py` | Contact, points, gates |
| `public/` | Canvas PPO opponent |

Code is MIT.
