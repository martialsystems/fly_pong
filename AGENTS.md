# Agent notes: fly_pong

MIT. GraphForge pin in `pongforge/`: phase A before B before C; lag 40/40 or PPO 99/100 is not a finished title; 100% vs a live copy is not skill. Verify-before-done is the finish gate.

The fly loop is the experiment. PPO in `public/` is a separate controller. Do not import T4/T5 from `public/`. Do not train T4/T5 as a PPO policy. Train gates and readouts only.

Locked rates: `logs/fly_gate.json`, `logs/fly_gate_motion_only.json`, `logs/device_move.json`, `logs/device_aim.json`, `logs/aim_hypothesis.json`, `logs/aim_vs_returner.json`, `logs/aim_sign_hypothesis.json`, `logs/aim_sign.json`. README quotes those files. Self-play stays locked until `aim_sign.json` has `selfplay: unlocked`. Do not retcon the +52 point OR-bar into placement. Do not train more vs the lag bot to force a title.

Physics lives in `shared/constants.json` plus `fly_pong/physics.py`. The browser clone is `public/js/physics.js`. After changing constants, run `python scripts/sync_constants.py`.

The website AI is a left-trained PPO policy. The right paddle must send a mirrored 6-D observation, then VecNormalize from `public/models/norm.json`.

Verify:

```
python3 ~/agent_laws_verify_before_done/vbd_gate.py check --app-root . --claim-done
```

Phone-width and desktop: `python scripts/viewport_sanity.py` (CDP device metrics, unique Chrome user-data-dir).
