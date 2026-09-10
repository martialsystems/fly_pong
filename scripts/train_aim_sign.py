#!/usr/bin/env python3
"""Supervised sign(u_offset) == sign(gap) vs frozen Phase A. Window stays 24."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fly_pong.constants import load_constants
from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_N
from fly_pong.physics import initial_state, step as physics_step
from fly_pong.play_device import _info_state, _mirror_right
from pongforge.gate import require_can_train_aim

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def train_epoch(
    device: FlyPongDevice,
    returner: FlyPongDevice,
    opt: torch.optim.Optimizer,
    seed: int,
    episodes: int,
) -> float:
    C = load_constants()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(episodes):
        state = initial_state(
            C,
            serve_dir=float(rng.choice(np.array([-1.0, 1.0]))),
            angle=float(rng.uniform(-C["serveAngleMax"], C["serveAngleMax"])),
        )
        device.reset()
        returner.reset()
        acc = []
        frames = 0
        while frames < 4000 and not state["terminated"]:
            info = _info_state(state, C)
            cmd = device.step_command(info)
            u = device.aim_router.command(cmd["bank"].aim)
            gap = float(cmd["bank"].aim[-1])  # desired_offset
            if cmd["bank"].incoming and abs(gap) > 0.02:
                gap_sign = 1.0 if gap > 0 else -1.0
                mag = 0.25 * (1.0 - torch.tanh(u.abs())).pow(2)
                acc.append(F.softplus(-gap_sign * u) + mag)
            opp_cmd = returner.step_command(_mirror_right(state, C))
            state, _pt = physics_step(state, float(cmd["dy"]), float(opp_cmd["dy"]), C, serve_angle=0.0)
            frames += 1
        if not acc:
            continue
        loss = torch.stack(acc).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--src", type=Path, default=ARTIFACTS / "device.pt")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "device_aim_sign.pt")
    args = parser.parse_args()

    require_can_train_aim()
    device = FlyPongDevice(aim_n=AIM_N)
    returner = FlyPongDevice(aim_n=0)
    if args.src.is_file():
        device.load(args.src, aim=True)
        returner.load(args.src, aim=False)
    returner.aim_n = 0
    for p in device.parameters_move():
        p.requires_grad = False
    opt = torch.optim.Adam(device.parameters_aim(), lr=0.08)
    for e in range(int(args.epochs)):
        mean_loss = train_epoch(
            device, returner, opt, seed=args.seed + e * 40, episodes=int(args.episodes)
        )
        print(f"epoch {e+1}/{args.epochs} loss={mean_loss:.4f} g_aim={device.aim_router.gates_np()}")
    device.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
