# Agent notes: fly_pong

MIT. No GraphForge pin. Two products in one tree: a Gymnasium trainer and a static browser opponent. The T4/T5 loop is Python-only. Do not import it from `public/`.

Physics lives in `shared/constants.json` plus `fly_pong/physics.py`. The browser clone is `public/js/physics.js`. After changing constants, run `python scripts/sync_constants.py`. A parity test steps both.

The website AI is a left-trained PPO policy. The right paddle must send a mirrored 6-D observation, then VecNormalize from `public/models/norm.json`.

Verify:

```
python3 ~/agent_laws_verify_before_done/vbd_gate.py check --app-root . --claim-done
```

Phone-width and desktop: `python scripts/viewport_sanity.py` (CDP device metrics, unique Chrome user-data-dir).
