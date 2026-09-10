# fly_pong

A 32-ommatidia T4/T5 strip pointed at Pong, plus a retinotopic centering reflex.

Motion alone cannot play. Centering can track and win against the env lag paddle.
Placement is closed. The browser court is a separate 6-input PPO, not the fly.

**Question.** Can a fly-style vertical motion detector play Pong?

**Answer.** Motion alone cannot. T4/T5 on a 32-ommatidium strip, 20 games vs the env lag paddle, seed 0: **0/20 matches, 0-220 points** (`logs/fly_gate_motion_only.json`). A still or sideways ball barely moves on a 1-D retina, so the paddle never acquires it.

Add a retinotopic centering reflex (luminance center of mass vs paddle height) and the same body won **40/40 matches, 439-46 points** at n=40, seed 0 (`logs/fly_gate.json`). One of those 40 hit the 20,000-frame cap at 10-4; the other 39 finished first to 11. Mean tracking error 0.023 of court height. That lock is vs the lag paddle (0.75× speed) built into `FlyPongEnv`.

That is tracking, not placement. Phase B vs lag is a null (open-hit ~0.5). Later aim heads moved points, geo, or contact against a frozen Phase A returner; none cleared geo, leak, and match bar together. Self-play is not part of this title.

Playable court: [martialsystems.github.io/fly_pong](https://martialsystems.github.io/fly_pong/). You are the left paddle. The right paddle is a 6-input PPO exported to ONNX. Same physics as the lab env, different controller. 99/100 on that net is not a fly result and not a placement result.

## Locked numbers

Copied from `logs/`. Lag 40/40 and PPO 99/100 are those evals only.

### Hand loop

`logs/fly_gate_motion_only.json`, `logs/fly_gate.json`

| Controller | n | Matches | Points | Track err |
|------------|--:|--------:|-------:|----------:|
| T4/T5 motion only | 20 | 0/20 | 0-220 | 0.274 |
| T4/T5 + centering | 40 | 40/40 | 439-46 | 0.023 |

### Device vs lag

`logs/device_move.json`, `logs/device_aim.json`. n=40, seed 0. Named channels, softmax gates, T4/T5 frozen. MoveRouter every frame. AimRouter only in the last 24 frames before contact. Bounce is still the geometric paddle offset in `physics.py`.

| Phase | Matches | Points | Contact | Open-hit | Biggest gate |
|-------|--------:|-------:|--------:|---------:|--------------|
| A move | 40/40 | 422-48 | 0.992 | 0.520 | error_y 0.984 |
| B aim vs lag | 40/40 | 420-47 | 0.992 | 0.497 | predicted_contact_t 0.900 |

Phase A mastered tracking via error_y. Phase B vs lag did not beat chase at placement. The clock channel `predicted_contact_t` is out of the aim bank (`logs/aim_hypothesis.json`).

### Aim vs a frozen Phase A returner

Same 40 seeds. Hypothesis files were written before the matching eval.

| Run | Log | Result |
|-----|-----|--------|
| Move-only | `logs/aim_vs_returner.json` | 12/40, 264-289, open-hit 0.531 |
| Move+aim | `logs/aim_vs_returner.json` | 18/40, 316-321, open-hit 0.568 |
| Signed-open | `logs/aim_sign.json` | sign(geo) vs open side 0.469 → 0.542 (Δ +0.073, bar +0.10). Open-hit Δ +0.037 (bar +0.08). Fail. |
| Supervised sign | `logs/aim_sign_supervised.json` | signed-open geo 0.469 → 0.814. Open-hit 0.531 → 0.701. Command 1.00, leak 0.186 (cap 0.08). Geo bar hit, leak cap missed. Matches 12/40 → 10/40. Fail. |
| Leak autopsy | `logs/aim_leak.json` | 1000/5372 leak contacts: 950 control_late, 47 inbound-vy, 3 wall-before-landing. Control share 0.95, physics share 0.05. |
| Setpoint | `logs/aim_setpoint.json` | Freeze intercept Y at window open, bang-bang in 24 frames. Leak 0.179, geo 0.821, matches 7/40 (bar 13). Fail leak and scoring. Geo held. Leak is late arrival at the commanded Y (0.95), not walls or inbound vy. |
| Commit-if-reachable | `logs/landing_commit.json` | Incoming and reach ≤ tau, then bang-bang to intercept Y, else error_y. Leak 0.000, geo 0.465, matches 17/40, points 211-204 vs this eval's move-only 11/40, 262-291. Contact recovered. Placement did not. |

Point delta +52 (316-321 vs 264-289) is not placement. Open-hit stayed under the bar (+0.037). |geo| 0.78 with signed-open 0.54 is a random corner-smasher: aim is coupled to bounce and can buy points without placing the ball.

Landing is contact. Contact was already solved. The commit gate is incoming and reach ≤ tau, then intercept Y; otherwise error_y.

### Unused parts as features and gates

Hypothesis files written first. Cheap filters on the court state and 1-D strip. No NeuroArch query. No Neurokernel. CX_heading and MB_value are traces in the eval dumps, not wired into dy or offset.

| Run | Log | Result |
|-----|-----|--------|
| Loom commit/abort | `logs/loom_commit.json` | Same reach-gated intercept Y as landing_commit, named LPLC tau. Vs frozen Phase A, n=40. Leak 0.000, contact 0.957, matches 17/40, points 261-260 vs move-only 11/40, 262-291, contact 0.951. Window 24. Offset off. Pass as a goalie veto. |
| Unused move gates | `logs/unused_move_gates.json` | Channels 1-7 on a separate move bank. error_y mass 0.973. Unused vision lost to error_y. Vs lag 40/40, 439-61 (chase, not a title). Vs Phase A 23/40, 283-277: a different training seed with error_y at 0.973; not unused-vision skill. Next unused mass is LC11_dark 0.011. Fail the pre-registered error_y bar. |

`logs/loom_commit.json` 17/40 and `logs/landing_commit.json` 17/40 are the same kind of object: reach-gated intercept Y. One veto, two names.

error_y still owns move. Loom is an abort/commit on reach. Unused fly-named filters are unique labels on court/strip features, not a second retina. Unused parts retargeted; plant unchanged; titles unchanged.

### Browser PPO

PPO on the 6-number box, 100 games: 99/100 (`public/models/pong.onnx`). That net does not share weights with the fly loop or the device.

## What the controllers are

Hand loop: photoreceptor strip → L1/L2 half-wave → Reichardt T4c/d and T5c/d → centering.

Device: the same strip plus `error_y`, time-to-paddle, LC-like blob energy, opponent open space, `desired_offset`. Two softmax routers. Commit gate in `fly_pong/commit.py` (reach ≤ tau; intercept Y, not offset).

Physics lives in `shared/constants.json` plus `fly_pong/physics.py`. The browser clone is `public/js/physics.js`. After changing constants, run `scripts/sync_constants.py`. Python and the page use the same step.

`pongforge/` encodes phase order and claim bans (no aim-before-move; lag 40/40 is not a finished title).

`fly_pong/fbl_adapter.py` is a Neurokernel stub. Public FFBO servers do not execute circuits.

## Reproduce

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
PYTHONPATH=. .venv/bin/python scripts/train_aim_sign.py
PYTHONPATH=. .venv/bin/python scripts/eval_aim_sign_supervised.py
PYTHONPATH=. .venv/bin/python scripts/eval_aim_leak.py
PYTHONPATH=. .venv/bin/python scripts/eval_aim_setpoint.py
PYTHONPATH=. .venv/bin/python scripts/eval_landing_commit.py
PYTHONPATH=. .venv/bin/python scripts/eval_loom_commit.py
PYTHONPATH=. .venv/bin/python scripts/train_unused_move.py
PYTHONPATH=. .venv/bin/python scripts/eval_unused_move_gates.py
.venv/bin/python -m http.server 8000 --directory public
```

Local court: http://127.0.0.1:8000
Published court: https://martialsystems.github.io/fly_pong/

After changing `public/`, publish with `scripts/publish_pages.sh`. Restamp tables from the JSON if the numbers move.

## Layout

| Path | Role |
|------|------|
| `logs/fly_gate.json` | Centering lock, n=40 |
| `logs/fly_gate_motion_only.json` | Motion-only lock, n=20 |
| `logs/device_move.json` | Phase A lock |
| `logs/device_aim.json` | Phase B vs lag (null placement) |
| `logs/aim_hypothesis.json` | Pass/fail written before the returner eval |
| `logs/aim_vs_returner.json` | Move-only vs move+aim vs Phase A |
| `logs/aim_sign_hypothesis.json` | Signed-open pass/fail, written first |
| `logs/aim_sign.json` | Signed-open vs Phase A |
| `logs/aim_sign_supervised_hypothesis.json` | Supervised-sign pass/fail, written first |
| `logs/aim_sign_supervised.json` | Geo bar hit, leak cap missed |
| `logs/aim_leak_hypothesis.json` | Leak autopsy pass/fail, written first |
| `logs/aim_leak.json` | 1000 leak rows; control 0.95, physics 0.05 |
| `logs/aim_setpoint_hypothesis.json` | Setpoint pass/fail, written first |
| `logs/aim_setpoint.json` | Freeze-Y in window: leak 0.179, 7/40 |
| `logs/landing_commit_hypothesis.json` | Commit veto pass/fail, written first |
| `logs/landing_commit.json` | Reachable lunge to intercept Y: leak 0, 17/40, geo 0.465 |
| `logs/loom_commit_hypothesis.json` | Loom veto pass/fail, written first |
| `logs/loom_commit.json` | Loom commit vs Phase A: leak 0, 17/40 |
| `logs/unused_move_gates_hypothesis.json` | Unused move-gate pass/fail, written first |
| `logs/unused_move_gates.json` | error_y 0.973; unused vision lost |
| `fly_pong/features.py` | Named move/aim channels plus unused 1-7 |
| `fly_pong/routers.py` | Softmax gates |
| `fly_pong/device.py` | Encode + two heads + commit veto |
| `fly_pong/commit.py` | Reach ≤ tau; intercept Y, not offset |
| `pongforge/` | Phase order and claim bans |
| `scripts/eval_fly.py` | Hand-loop match rate |
| `scripts/eval_device.py` | Contact, points, gates |
| `scripts/eval_landing_commit.py` | Commit veto vs Phase A freeze |
| `scripts/eval_loom_commit.py` | Loom veto vs Phase A freeze |
| `scripts/train_unused_move.py` | Refit move gates over unused 1-7 |
| `scripts/eval_unused_move_gates.py` | Unused gates vs lag and Phase A |
| `public/` | Canvas 6-D PPO opponent (not the fly loop) |
| `scripts/publish_pages.sh` | Copy `public/` onto `gh-pages` |

Sequel work needs a new question. MIT license.
