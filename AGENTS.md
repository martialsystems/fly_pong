# Agent notes: fly_pong

MIT. No GraphForge pin. The fly loop is the experiment. PPO in `public/` is a separate controller. Do not import T4/T5 from `public/`.

Locked match rates live in `logs/fly_gate.json` and `logs/fly_gate_motion_only.json`. README quotes those files. `scripts/eval_fly.py` rewrites the lock. Restamp the README table from the JSON.

Physics lives in `shared/constants.json` plus `fly_pong/physics.py`. The browser clone is `public/js/physics.js`. After changing constants, run `python scripts/sync_constants.py`.

The website AI is a left-trained PPO policy. The right paddle must send a mirrored 6-D observation, then VecNormalize from `public/models/norm.json`.

Verify:

```
python3 ~/agent_laws_verify_before_done/vbd_gate.py check --app-root . --claim-done
```

Phone-width and desktop: `python scripts/viewport_sanity.py` (CDP device metrics, unique Chrome user-data-dir).
