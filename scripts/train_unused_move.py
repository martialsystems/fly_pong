#!/usr/bin/env python3
"""EXP 2: refit move gates over unused vision channels. Not a lag title."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from fly_pong.constants import load_constants
from fly_pong.device import UnusedMoveDevice
from fly_pong.features import MOVE_KEYS, UNUSED_MOVE_KEYS
from fly_pong.physics import dy_from_action, initial_state, lag_opponent_dy, step as physics_step
from fly_pong.play_device import _info_state

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _action_from_u(u: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
    logits = torch.stack([torch.zeros((), dtype=u.dtype) + 0.15, -u, u])
    dist = Categorical(logits=logits)
    return dist, dist.sample()


def train_epoch(device: UnusedMoveDevice, opt: torch.optim.Optimizer, seed: int, episodes: int) -> float:
    C = load_constants()
    rng = np.random.default_rng(seed)
    total = 0.0
    device.move_router.train()
    for _ep in range(episodes):
        state = initial_state(
            C,
            serve_dir=float(rng.choice(np.array([-1.0, 1.0]))),
            angle=float(rng.uniform(-C["serveAngleMax"], C["serveAngleMax"])),
        )
        device.reset()
        logps = []
        rewards = []
        frames = 0
        while frames < 4000 and not state["terminated"]:
            info = _info_state(state, C)
            bank = device.encoder.encode(info)
            feats = np.concatenate([bank.move, bank.unused], axis=0)
            u = device.move_router.command(feats)
            dist, a = _action_from_u(u)
            logps.append(dist.log_prob(a))
            dy = dy_from_action(int(a.item()), C=C)
            opp_dy = lag_opponent_dy(state, C)
            prev_vx = float(state["ball_vx"])
            state, _pt = physics_step(state, dy, opp_dy, C, serve_angle=0.0)
            ph = float(C["paddleH"])
            err = abs((state["agent_y"] + ph / 2.0) - state["ball_y"]) / float(C["height"])
            r = 0.0
            if prev_vx < 0:
                r -= err
            if prev_vx < 0 and float(state["ball_vx"]) > 0:
                r += 1.0
            if _pt < 0:
                r -= 1.0
            if _pt > 0:
                r += 0.2
            rewards.append(r)
            frames += 1
        if not logps:
            continue
        ret = 0.0
        rets = []
        for r in reversed(rewards):
            ret = r + 0.99 * ret
            rets.append(ret)
        rets.reverse()
        adv = torch.tensor(rets, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = -(torch.stack(logps) * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(adv.mean())
    return total / max(episodes, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "device_unused_move.pt")
    args = parser.parse_args()

    device = UnusedMoveDevice()
    opt = torch.optim.Adam(device.parameters_move(), lr=0.08)
    for e in range(int(args.epochs)):
        train_epoch(device, opt, seed=args.seed + e * 100, episodes=int(args.episodes))
        g = device.move_router.gates_np()
        names = MOVE_KEYS + UNUSED_MOVE_KEYS
        mass = {k: float(g[i]) for i, k in enumerate(names)}
        print(f"epoch {e+1}/{args.epochs} error_y={mass['error_y']:.3f}")
    device.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
