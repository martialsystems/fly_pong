# Agent notes: fly_pong

MIT. GraphForge pin in `pongforge/`: phase A before B before C; lag 40/40 or PPO 99/100 is not a finished title; 100% vs a live copy is not skill. Verify-before-done is the finish gate.

The fly loop is the experiment (`run_fly`: 32-ommatidia strip → T4/T5 → centering, then aim/landing gates). PPO in `public/` is a separate controller. The court page is a toy baseline on the same physics. It is not the fly circuit. 99/100 is not a placement result. Do not import T4/T5 from `public/`. Do not train T4/T5 as a PPO policy. Train gates and readouts only.

Do not start another aim head on this object. Contact is readable. Placement is not. Self-play remains locked.

Do not restamp lag 40/40 or PPO 99/100 onto placement. Do not retcon the +52 point OR-bar into aim. Point delta +52 vs the freeze is not placement. Open-hit stayed under the bar. Do not stamp lag 40/40 or PPO 99/100 from `logs/approach_commit.json`. That file is a valence flip, not a fly result and not a placement result.

Locked rates: `logs/fly_gate.json`, `logs/fly_gate_motion_only.json`, `logs/device_move.json`, `logs/device_aim.json`, `logs/aim_hypothesis.json`, `logs/aim_vs_returner.json`, `logs/aim_sign_hypothesis.json`, `logs/aim_sign.json`. README quotes measured outcomes from those files. Self-play stays locked. Do not widen the aim window past 24. Do not write a C hypothesis while leak > 0.08 or matches < 13/40 vs the freeze, and do not write one from a geo miss. Setpoint lock: `logs/aim_setpoint.json`. Commit veto lock: `logs/landing_commit.json`. Loom commit lock: `logs/loom_commit.json`. Unused move-gate lock: `logs/unused_move_gates.json`. Approach-commit lock: `logs/approach_commit.json`. Landing is contact (intercept Y when reach ≤ tau). It is not an aim title. `logs/loom_commit.json` 17/40 and `logs/landing_commit.json` 17/40 are one veto, two names. Do not count them as two independent beat-12/40 results. Do not write C from 17/40 plus leak 0: geo/offset stayed off; contact was already solved. Do not write C from approach_commit 24/40 plus leak 0: contact was the bar; open-hit stayed ~0.53. Do not confuse `approach_commit` with `landing_commit`. Do not promote LC11_dark 0.011 into a third exp. Do not train another unused bank vs lag to force a title. Do not train more vs the lag bot to force a title. Do not wire CX_heading or MB_value into paddle dy or offset. Do not change `public/`. Do not lengthen the 24-frame window.

Hypothesis log lines (process, not README voice):

- Phase A mastered tracking via error_y. Phase B vs lag did not beat chase at placement (open-hit ~0.5). Self-play stays locked until aim beats move-only against something that returns the ball.
- Aim is coupled to bounce (|geo| 0.78) and buys points vs a tracker (+52) without clearing placement (open-hit +0.037). Self-play remains locked.
- Signed-open missed both bars. Command 0.65 vs geo 0.54 is leak, not a title. Next is supervised sign with a geo bar and a leak cap; miss that and aim is retired as a head.
- Supervised sign raised geo signed-open to 0.814 and open-hit to 0.701; leak 0.186 and matches 10/40 failed the AND. Aim stays a head. Self-play stays locked. Next is a leak autopsy, not a physics rewrite dressed as training.
- Leak is late arrival at the commanded Y (0.95), not walls or inbound vy. Next is setpoint-on-aim-Y in the existing window; physics stays frozen; self-play stays locked.
- Landing is a commit gate: incoming and reach <= tau, then bang-bang to intercept Y; else error_y. Contact, not placement. Same bars as the setpoint run. No C file. Self-play stays locked.
- Loom commit/abort: incoming and reach <= tau, bang-bang to Y_pred, else error_y. Window 24. Pass is a goalie veto, not a title. Self-play stays locked.
- Unused vision channels 1-7 as move gates only. If error_y mass stays >= 0.8, unused vision lost to error_y. Lag 40/40 is chase. Vs Phase A, matches must beat 12/40 to count as useful. Self-play stays locked.
- Valence flip: same 24-frame loom gate, ESCAPE_SIGN -1 occupies the crossing Y (approach), +1 flees. Contact, not placement. Open-hit is not a success metric. Not a fly result. Self-play stays locked.

Physics lives in `shared/constants.json` plus `fly_pong/physics.py`. The browser clone is `public/js/physics.js`. After changing constants, run `python scripts/sync_constants.py`. After changing `public/`, publish with `scripts/publish_pages.sh`.

The website opponent is a left-trained PPO policy. The right paddle must send a mirrored 6-D observation, then VecNormalize from `public/models/norm.json`. Label it as PPO, not as the fly.

Verify:

```
python3 ~/agent_laws_verify_before_done/vbd_gate.py check --app-root . --claim-done
```

Phone-width and desktop: `python scripts/viewport_sanity.py` (CDP device metrics, unique Chrome user-data-dir).
