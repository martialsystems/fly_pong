#!/usr/bin/env python3
"""Phase B: train AimRouter. Requires phase A contact bar."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from fly_pong.constants import load_constants
from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_KEYS
from fly_pong.physics import initial_state, lag_opponent_dy, step as physics_step
from fly_pong.play_device import _info_state
from pongforge.gate import require_can_train_aim

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def train_epoch(device: FlyPongDevice, opt: torch.optim.Optimizer, seed: int, episodes: int) -> None:
    C = load_constants()
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        state = initial_state(
            C,
            serve_dir=float(rng.choice(np.array([-1.0, 1.0]))),
            angle=float(rng.uniform(-C["serveAngleMax"], C["serveAngleMax"])),
        )
        device.reset()
        logps = []
        rewards = []
        frames = 0
        ph = float(C["paddleH"])
        h = float(C["height"])
        while frames < 4000 and not state["terminated"]:
            info = _info_state(state, C)
            cmd = device.step_command(info)
            u_off = device.aim_router.command(cmd["bank"].aim)
            dist = Normal(u_off, 0.15)
            sample = dist.rsample()
            offset = torch.clamp(sample, -1.0, 1.0)
            target = torch.tensor(
                float(cmd["bank"].aim[AIM_KEYS.index("desired_offset")]),
                dtype=torch.float32,
            )
            sup = (u_off - target) ** 2
            if cmd["aim_active"]:
                logps.append(dist.log_prob(sample) - 0.5 * sup)
                target_center = cmd["bank"].predicted_contact_y_px - float(offset.detach()) * (ph / 2.0)
                target_y = target_center - ph / 2.0
                err = target_y - state["agent_y"]
                speed = float(C["paddleSpeed"])
                if err > 1:
                    dy = speed
                elif err < -1:
                    dy = -speed
                else:
                    dy = 0.0
            else:
                dy = float(cmd["dy"])
                logps.append(-0.25 * sup)
            opp_dy = lag_opponent_dy(state, C)
            prev_vx = float(state["ball_vx"])
            open_down = (h - (state["opp_y"] + ph)) - state["opp_y"]
            state, pt = physics_step(state, dy, opp_dy, C, serve_angle=0.0)
            r = 0.0
            if prev_vx < 0:
                r -= abs((state["agent_y"] + ph / 2.0) - state["ball_y"]) / h * 0.2
            if prev_vx < 0 and float(state["ball_vx"]) > 0:
                r += 0.3
                if open_down > 0 and state["ball_vy"] > 0:
                    r += 1.0
                elif open_down < 0 and state["ball_vy"] < 0:
                    r += 1.0
            if pt > 0:
                r += 0.5
            if pt < 0:
                r -= 0.5
            rewards.append(r)
            frames += 1
        if not logps or not rewards:
            continue
        n = min(len(logps), len(rewards))
        ret = 0.0
        rets = []
        for r in reversed(rewards[:n]):
            ret = r + 0.99 * ret
            rets.append(ret)
        rets.reverse()
        adv = torch.tensor(rets, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = -(torch.stack(logps[:n]) * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weights", type=Path, default=ARTIFACTS / "device.pt")
    args = parser.parse_args()

    require_can_train_aim()
    device = FlyPongDevice()
    if args.weights.is_file():
        device.load(args.weights)
    for p in device.parameters_move():
        p.requires_grad = False
    opt = torch.optim.Adam(device.parameters_aim(), lr=0.05)
    for e in range(int(args.epochs)):
        train_epoch(device, opt, seed=args.seed + e * 50, episodes=int(args.episodes))
        print(f"epoch {e+1}/{args.epochs} g_aim={device.aim_router.gates_np()}")
    device.save(args.weights)
    print(f"saved {args.weights}")


if __name__ == "__main__":
    main()
