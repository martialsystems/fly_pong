# fly_pong

Closed as a placement title. A 32-ommatidia centering tracker in the lab, and a 6-input PPO on the court page.

**Question.** Can a fly-style vertical motion detector play Pong?

**Answer.** Motion alone cannot. T4/T5 on a 32-ommatidium strip, 20 games vs the env's lag paddle, seed 0: **0/20 matches, 0-220 points** (`logs/fly_gate_motion_only.json`). A still or sideways ball barely moves on a 1-D retina, so the paddle never acquires it.

Add a retinotopic centering reflex (luminance center of mass vs the paddle's height) and the same body won **40/40 matches, 439-46 points** at n=40, seed 0 (`logs/fly_gate.json`). One of those 40 hit the 20,000-frame cap at 10-4; the other 39 finished first to 11. Mean tracking error 0.023 of court height. That is a centering loop on a 32-pixel strip. The lock is vs the lag paddle (0.75× speed) built into `FlyPongEnv`.

Placement does not follow. Phase B vs lag is a null (open-hit ~0.5, clock on arrival). Signed-open, supervised sign, and a frozen intercept-Y setpoint all missed the AND. Commit-if-reachable is a landing veto, not an aim title: incoming and reach ≤ tau, then bang-bang to intercept Y, else error_y. Vs the same Phase A freeze it recovered contact (17/40, leak 0.000) and missed geo (0.465 vs 0.80). Landing is contact. Contact was already solved.

Play the PPO: [martialsystems.github.io/fly_pong](https://martialsystems.github.io/fly_pong/). Six numbers in, ONNX in the page.

Do not start another aim head on this object. Contact is readable. Placement is not. Self-play remains locked.

## Locked numbers

Copied from `logs/`. Do not restamp lag 40/40 or PPO 99/100 onto placement. Do not retcon the +52 point OR-bar into aim.

**Hand loop** (`logs/fly_gate_motion_only.json`, `logs/fly_gate.json`):

| Controller | n | Matches | Points | Track err |
|------------|--:|--------:|-------:|----------:|
| T4/T5 motion only | 20 | 0/20 | 0-220 | 0.274 |
| T4/T5 + centering | 40 | 40/40 | 439-46 | 0.023 |

**Device vs lag** (`logs/device_move.json`, `logs/device_aim.json`), n=40, seed 0. Named channels, softmax gates, T4/T5 frozen. MoveRouter every frame. AimRouter only in the last 24 frames before contact. Bounce is still the geometric paddle offset in `physics.py`.

| Phase | Matches | Points | Contact | Open-hit | Biggest gate |
|-------|--------:|-------:|--------:|---------:|--------------|
| A move | 40/40 | 422-48 | 0.992 | 0.520 | error_y 0.984 |
| B aim vs lag | 40/40 | 420-47 | 0.992 | 0.497 | predicted_contact_t 0.900 |

Phase A mastered tracking via error_y. Phase B vs lag did not beat chase at placement (open-hit ~0.5). Self-play stays locked until aim beats move-only against something that returns the ball. The clock channel `predicted_contact_t` is out of the aim bank. `logs/aim_hypothesis.json`.

**Vs a frozen Phase A returner**, same 40 seeds (`logs/aim_hypothesis.json` then `logs/aim_vs_returner.json`): move-only 12/40, 264-289, open-hit 0.531. Move+aim 18/40, 316-321, open-hit 0.568. Point delta +52 cleared a loose OR-bar. Open-hit +0.037 did not. That is not placement.

Signed-open eval (`logs/aim_sign_hypothesis.json` written first, then `logs/aim_sign.json`): sign(geo) vs open side 0.469 → 0.542 (Δ +0.073, bar +0.10). Open-hit Δ still +0.037 (bar +0.08). |geo| 0.78 with signed-open 0.54 is a random corner-smasher. Aim is coupled to bounce (|geo| 0.78) and buys points vs a tracker (+52) without clearing placement (open-hit +0.037). Self-play remains locked.

Signed-open missed both bars. Command 0.65 vs geo 0.54 is leak, not a title. Next is supervised sign with a geo bar and a leak cap; miss that and aim is retired as a head.

Supervised sign (`logs/aim_sign_supervised_hypothesis.json` then `logs/aim_sign_supervised.json`): `desired_offset` gate 0.994. Vs the same Phase A freeze, signed-open geo 0.469 → 0.814 (Δ +0.345, bar +0.10). Open-hit 0.531 → 0.701 (Δ +0.170). Command 1.00, leak cmd−geo 0.186 (cap 0.08). Written pass is both geo and leak: **fail**. Bounce map is not the limit (geo moved). Aim is not retired. Self-play stays locked. Match WR 12/40 → 10/40. C stays closed.

Leak autopsy (`logs/aim_leak_hypothesis.json` then `logs/aim_leak.json`), same 40 seeds, dump of the 1000/5372 leak contacts: 950 control_late (paddle_err > 7 px), 47 inbound-vy dominates offset, 3 wall-before-landing. Physics share 0.05, control share 0.95. Bounce law is not the cap. Landing dist 111. No C hypothesis. Supervised sign raised geo signed-open to 0.814 and open-hit to 0.701; leak 0.186 and matches 10/40 failed the AND. Aim stays a head. Self-play stays locked. Next is a leak autopsy, not a physics rewrite dressed as training.

Setpoint run (`logs/aim_setpoint_hypothesis.json` then `logs/aim_setpoint.json`): freeze intercept Y at window open, bang-bang in 24 frames. Leak 0.179 (bar 0.08), geo 0.821 (bar 0.80), matches 7/40 (bar 13). Fail leak and scoring. Geo held. Window not widened. No C file. Leak is late arrival at the commanded Y (0.95), not walls or inbound vy. Next is setpoint-on-aim-Y in the existing window; physics stays frozen; self-play stays locked.

Commit-if-reachable (`logs/landing_commit_hypothesis.json` then `logs/landing_commit.json`): incoming and reach ≤ tau, bang-bang to intercept Y, else error_y. Abort share in the 24-frame window 0.210. Leak 0.000 (bar 0.08), geo 0.465 (bar 0.80), matches 17/40 (bar 13), points 211-204 vs this eval's move-only 11/40, 262-291. Fail geo. Contact recovered from the 7/40 setpoint. Placement did not. No C file. Self-play stays locked. Landing is a commit gate: incoming and reach <= tau, then bang-bang to intercept Y; else error_y. Contact, not placement. Same bars as the setpoint run. No C file. Self-play stays locked.

PPO on the 6-number box, 100 games, was 99/100 (`public/models/pong.onnx`). That net does not share weights with the fly loop or the device.

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
.venv/bin/python -m http.server 8000 --directory public
```

Court page: https://martialsystems.github.io/fly_pong/

After changing `public/`, publish with `scripts/publish_pages.sh` (copies `public/` onto `gh-pages`).

Restamp tables from the JSON if the numbers move.

## Layout

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
| `logs/aim_sign_supervised_hypothesis.json` | Supervised-sign pass/fail, written first |
| `logs/aim_sign_supervised.json` | Geo bar hit, leak cap missed, C locked |
| `logs/aim_leak_hypothesis.json` | Leak autopsy pass/fail, written first |
| `logs/aim_leak.json` | 1000 leak rows; control 0.95, physics 0.05 |
| `logs/aim_setpoint_hypothesis.json` | Control-run pass/fail, written first |
| `logs/aim_setpoint.json` | Freeze-Y in window: leak 0.179, 7/40 |
| `logs/landing_commit_hypothesis.json` | Commit veto pass/fail, written first |
| `logs/landing_commit.json` | Reachable lunge to intercept Y: leak 0, 17/40, geo 0.465 |
| `fly_pong/features.py` | Named move/aim channels |
| `fly_pong/routers.py` | Softmax gates |
| `fly_pong/device.py` | Encode + two heads + commit veto |
| `fly_pong/commit.py` | Reach ≤ tau; intercept Y, not offset |
| `pongforge/` | Phase order and claim bans |
| `scripts/eval_fly.py` | Hand-loop match rate |
| `scripts/eval_device.py` | Contact, points, gates |
| `scripts/eval_landing_commit.py` | Commit veto vs Phase A freeze |
| `public/` | Canvas PPO opponent |
| `scripts/publish_pages.sh` | Copy `public/` onto `gh-pages` |

Gymnasium Pong. Physics in `shared/constants.json`. Python and the browser clone the same step.

Hand loop: photoreceptor strip → L1/L2 half-wave → Reichardt T4c/d and T5c/d → centering. Device: the same strip plus `error_y`, time-to-paddle, LC-like blob energy, opponent open space, `desired_offset`. Commit gate: `fly_pong/commit.py`. `pongforge/` refuses aim-before-move and refuses lag-as-finished-title claims.

`fly_pong/fbl_adapter.py` is a Neurokernel stub. Public FFBO servers do not execute circuits.

`public/` is a canvas opponent that loads the PPO ONNX.

Sequel work needs a new question. Code is MIT.
