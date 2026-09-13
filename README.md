# fly_pong

[PLAY HUMAN VS FLY](https://martialgames.net/fly-pong/) online now!

- **Left paddle:** W/S, arrows, or mouse. Right paddle: the 24-frame intercept goalie (fly). Press 'M' to mute, 'R' to reset.

**Or if you wish to download and test yourself:**

First clone: make `.venv` under [Reproduce](#reproduce).

```bash
.venv/bin/python -m fly_pong.run_human --opponent approach
```
# Q&A

**What’s Actually Happening Here?**

This project isn’t a full fruit-fly brain playing video games. Think of it as a focused lab experiment testing a simple question: *Can an insect’s built-in visual motion detectors play Pong?*

**The short answer:** Motion detection alone cannot hit the ball. What actually works is a simple visual reflex. The fly tracks the bright spot representing the ball and centers its paddle over it. That’s enough to beat a slow computer opponent, but it’s just basic tracking, not strategic play.

**How the Fly "Sees":**

The fly does not see a full Pong court with paddles and walls.

- **The Retina:** It sees the game through a single vertical strip of 32 sensory pixels.
- **The Image:** The ball shows up as a blurry, glowing dot on a dark background. The position of that dot is the only visual information the system gets.

## Phase 1: Motion Detection Alone (Why It Fails)

The first model relies purely on standard insect motion vision (T4/T5 circuits), which work by comparing pixel brightness frame-by-frame:

1. It checks if pixels are getting brighter or darker.

2. It delays one signal slightly to compare neighboring pixels.

3. If a bright spot shifts down, it tells the paddle to move down (and vice versa).

**The Problem:** A ball moving straight across the screen (or very slowly), creates almost no vertical motion on a 1-D strip. The paddle stands still and misses every shot. Result: **0 wins in 20 matches.**

## Phase 2: What Actually Works (The Centering Reflex)

To fix this, the authors added a second, position-based control loop:

1. **Find the Blob:** Calculate where the bright spot is on the 32-pixel retina.

2. **Find the Paddle:** Check the current paddle position (fed directly from game state, not vision).

3. **Close the Gap:** Move the paddle toward the bright spot.

4. **Boost on Approach:** When the ball flies toward the fly, it boosts movement sensitivity (a "looming" reaction). Motion data is added only as a minor predictive nudge.

**The Result:** By constantly trying to align its center with the ball's center, the fly won 40 out of 40 matches against a slowed-down baseline opponent. Rather than playing strategy, it acts as an  automated target-tracker.
## Advanced Layers (And the Limits of "Aiming")

**I also tested extra control layers:**

- **Interception Windows:** If a ball is incoming and reachable, the paddle snaps directly to the predicted impact point instead of drifting toward it.

- **Goalie Lunges:** Positioning the paddle to block the ball rather than moving away from it.

While these tweaks improved rally counts against simple opponents, none of them produced true, intentional "aiming" or strategic ball placement.

# What’s the Next Step Here?

Right now, the Pong game is just a downstream toy. Retuning arbitrary time constants until the paddle moves "like a biological fly" isn't neuroscience; it's just tweaking math to look cool. 

Motion-only is still blind to a still blob. The paddle starts 1–2 frames late, so vs you it should feel slightly drunk. That FIFO is the placeholder to delete when clocks live in the neurons.

Eval writers log motor_delay_frames, delay_ms_assumed, and t4t5_tau. I did not re-run or restamp lag 40/40.

If you want to build a truly biological model, you have to separate the fly’s vision from the arcade game. This comes down to a **three-layer roadmap**: fix the biological physics, upgrade the neural wiring, and (only if you're feeling ambitious) test it on a real animal.

## Layer 1: Biological Calibration (The Immediate Next Experiment)

Freeze the Pong game completely. Stop scoring whether the paddle hits the ball and start scoring how the simulated visual cells react to real laboratory light tests. 

Set up a biological clock in milliseconds, map the 32-pixel strip to a real fly's field of view (around 5° per lens, covering a 160° arc), and run the model through the exact visual stimuli neuroscientists use in live cell experiments:
- **Flashes:** Flash bright and dark bars for 10 to 50 ms. Real T4 cells react strictly to ON (brightening), while T5 cells react to OFF (darkening).
- **Apparent Motion:** Show two adjacent light flashes separated by precise delays (8 to 500 ms). Real motion-detecting cells peak around a 17 ms delay.
- **Moving Edges & Gratings:** Pass light/dark stripes across the vision strip at varying speeds (0.1 to 8 Hz). Real fly vision peaks at around 1 Hz, not high-speed 10 Hz movement.

Save these benchmark metrics into a structured log (`t4t5_physiology.json`). If the current math misses the 1 Hz frequency peak or the 17 ms correlation delay (which it probably will), **that’s a success**. It proves your cell test is working, whereas the Pong game would have hidden those biological flaws.

## Layer 2: Replacing Cartoon Math with Real Physiology

The current motion detector relies on a simplified two-arm multiplication (`Current Pixel × Delayed Neighbor`). Real T4 neurons are much more complex, using three inputs: a delayed preferred side, a fast center, and a delayed suppression side to stop reverse movement.

Instead of trying to simulate a massive, overwhelming brain map (connectome), make the smallest realistic upgrades:

1. **Realistic Filters:** Replace basic cutoffs (`max(x,0)`) with published biological filter kernels.
2. **Proper Delays:** Separate input delays into real-world physiological ranges (~13 to 20 ms).
3. **Slow-Arm Channels:** Add input channels in the 100 to 500 ms range to properly recreate the 1 Hz frequency preference.

Once those are added, rerun the Layer 1 test suite. Now you're checking if a model built with real millisecond kinetics actually acts like a biological cell.

## Layer 3: The Live Lab (Where Real Science Happens)

If you want true biological proof, you stop using video games entirely. You put a real, tethered fruit fly under a two-photon microscope in an arena, display the exact same visual flash/motion tests, and compare real neural signal traces against your code's traces.

At this level, nobody cares if a paddle hits a ball on a 60 fps digital screen. They care about matching biological data to software data.

---

To keep the project grounded, don't mix up what each test is actually measuring:

| Question | What You're Testing |
| :--- | :--- |
| **"Does our math filter twitch in a few frames?"** | The original Pong code latency test. |
| **"Does our filter behave like real fly cells?"** | **Layer 1:** Milliseconds and degrees benchmark. |
| **"Does a biologically accurate T4 model still hit the ball?"** | Only test this *after* Layer 1 passes. |
| **"Can an actual fly play Pong?"** | **Layer 3:** Real living animal + lab setup (Almost certainly no). |

# Technical Bits:

- A 32-ommatidia T4/T5 strip pointed at Pong, plus a retinotopic centering reflex.

- Motion alone cannot play. Centering can track and win against the env lag paddle.
Placement is closed. A 1-D motion strip can return a Pong ball. It does not place one.

**Question.** Can a fly-style vertical motion detector play Pong?

**Answer.** Motion alone cannot. T4/T5 on a 32-ommatidium strip, 20 games vs the env lag paddle, seed 0: **0/20 matches, 0-220 points** (`logs/fly_gate_motion_only.json`). A still or sideways ball barely moves on a 1-D retina, so the paddle never acquires it.

Add a retinotopic centering reflex (luminance center of mass vs paddle height) and the same body won **40/40 matches, 439-46 points** at n=40, seed 0 (`logs/fly_gate.json`). One of those 40 hit the 20,000-frame cap at 10-4; the other 39 finished first to 11. Mean tracking error 0.023 of court height. That lock is vs the lag paddle (0.75× speed) built into `FlyPongEnv`.

That is tracking, not placement. Phase B vs lag is a null (open-hit ~0.5). Later aim heads moved points, geo, or contact against a frozen Phase A returner; none cleared geo, leak, and match bar together. Self-play is not part of this title.

The playable window uses bounceGain 1.008 with no speed cap, so the ball keeps climbing until the fly's 24-frame reach gate fails. Locked evals keep bounceGain 1.02. This court is not a fly result and not a placement result.

## Locked numbers

Copied from `logs/`. Lag 40/40 is chase.

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

### Valence flip (approach commit)

`logs/approach_commit_hypothesis.json` written first. Result: `logs/approach_commit.json`.

Same 24-frame loom gate as landing/loom. `ESCAPE_SIGN = -1` occupies the predicted crossing Y (approach). `ESCAPE_SIGN = +1` flees. Default is the existing centering reflex. Hand loop is unchanged: photoreceptor strip → L1/L2 → T4/T5 → centering. The new code is a commit/motor remap.

Hypothesis: sign-flipping loom from leave-the-point to occupy-the-crossing-Y yields contact ≥ landing_commit and is just a goalie, not placement. Open-hit is not a success metric.

Bakeoff vs the env lag paddle, n=40, seed 0, same seeds (`play_match` RNG; the fly_gate lock used `eval_fly` / Gym seeding).

| Controller | Matches | Points | Contact | Track err | Commit frac |
|------------|--------:|-------:|--------:|----------:|------------:|
| T4/T5 motion only | 0/40 | 11-440 | 0.460 | 0.179 | 0 |
| T4/T5 + centering | 40/40 | 437-61 | 0.987 | 0.023 | 0 |
| approach_commit (32-ommatidia Y) | 40/40 | 427-25 | 0.996 | 0.026 | 0.185 |
| approach_commit oracle Y | 40/40 | 429-31 | 0.995 | 0.025 | 0.189 |

Lag 40/40 on this remap is chase, the same family as centering. This is a valence flip, not a fly result and not a placement result.

Vs frozen Phase A, n=40, seed 0 (landing/loom opponent). Pass key: contact ≥ landing_commit 0.968.

| Run | Matches | Points | Contact | Leak | Open-hit | \|error_y\| at commit | Reach miss |
|-----|--------:|-------:|--------:|-----:|---------:|----------------------:|-----------:|
| landing_commit | 17/40 | 211-204 | 0.968 | 0 | 0.520 | | |
| loom_commit | 17/40 | 261-260 | 0.957 | 0 | 0.511 | | |
| approach_commit | 24/40 | 229-169 | 0.974 | 0 | 0.532 | 0.062 | 0.089 |
| approach_commit oracle Y | 26/40 | 230-168 | 0.974 | 0 | 0.536 | 0.062 | 0.090 |

Contact 0.974 ≥ 0.968. Leak 0. Mean |error_y| at commit 0.044 vs lag, 0.062 vs Phase A. Commit fraction 0.185 vs lag, 0.175 vs Phase A. Oracle Y matched the strip estimate at contact. Open-hit stayed ~0.53. Offset off. Window 24. Pass as a goalie valence flip. Matches vs the freeze are reported; they are not a placement bar.

## What the controllers are

Hand loop: photoreceptor strip → L1/L2 half-wave → Reichardt T4c/d and T5c/d → centering.

Device: the same strip plus `error_y`, time-to-paddle, LC-like blob energy, opponent open space, `desired_offset`. Two softmax routers. Commit gate in `fly_pong/commit.py` (reach ≤ tau; intercept Y, not offset). `approach_commit` is that gate with `ESCAPE_SIGN` flipped to occupy the crossing Y.

Physics lives in `shared/constants.json` plus `fly_pong/physics.py`. The playable court is `fly_pong/court.py` plus `run_human`.

`pongforge/` encodes phase order and claim bans (no aim-before-move; lag 40/40 is not a finished title).

`fly_pong/fbl_adapter.py` is a Neurokernel stub. Public FFBO servers do not execute circuits.

## Reproduce

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,train]"
.venv/bin/python -m pytest
```

Do not use stock `/usr/bin/python3 -m pytest`.

```bash
.venv/bin/python -m fly_pong.run_human --opponent approach
.venv/bin/python -m fly_pong.run_human --opponent lag
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
PYTHONPATH=. .venv/bin/python scripts/eval_approach.py --games 40 --seed 0 --out logs/approach_commit.json --oracle_y
PYTHONPATH=. .venv/bin/python scripts/train_unused_move.py
PYTHONPATH=. .venv/bin/python scripts/eval_unused_move_gates.py
```

Restamp tables from the JSON if the numbers move.

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
| `logs/approach_commit_hypothesis.json` | Valence-flip pass/fail, written first |
| `logs/approach_commit.json` | Approach commit: contact 0.974 vs Phase A; lag chase is not a title |
| `logs/unused_move_gates_hypothesis.json` | Unused move-gate pass/fail, written first |
| `logs/unused_move_gates.json` | error_y 0.973; unused vision lost |
| `fly_pong/features.py` | Named move/aim channels plus unused 1-7 |
| `fly_pong/routers.py` | Softmax gates |
| `fly_pong/device.py` | Encode + two heads + commit veto |
| `fly_pong/commit.py` | Reach ≤ tau; intercept Y, not offset. `ESCAPE_SIGN` approach/flee |
| `pongforge/` | Phase order and claim bans |
| `scripts/eval_fly.py` | Hand-loop match rate |
| `scripts/eval_device.py` | Contact, points, gates |
| `scripts/eval_landing_commit.py` | Commit veto vs Phase A freeze |
| `scripts/eval_loom_commit.py` | Loom veto vs Phase A freeze |
| `scripts/eval_approach.py` | Valence-flip bakeoff; `--oracle_y` reports true Y too |
| `scripts/train_unused_move.py` | Refit move gates over unused 1-7 |
| `scripts/eval_unused_move_gates.py` | Unused gates vs lag and Phase A |
| `fly_pong/court.py` | Medieval hall renderer |
| `fly_pong/run_human.py` | human vs fly (or lag) |

MIT. [Fly research index](https://gist.github.com/martialsystems/12835f747d6360781f3cc7f91f243178)
