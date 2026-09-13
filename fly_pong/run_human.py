"""Human left paddle vs lag or approach_commit. Python court only."""

from __future__ import annotations

import argparse
from typing import Any

from fly_pong.commit import ApproachCommitController
from fly_pong.constants import load_constants
from fly_pong.court import draw as draw_court
from fly_pong.env import FlyPongEnv
from fly_pong.physics import clip, lag_opponent_dy, mirror_right_state


def human_dy(keys, mouse_y: float, agent_y: float, C: dict[str, Any], pygame) -> float:
    speed = float(C["paddleSpeed"])
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return -speed
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return speed
    target = clip(float(mouse_y) - float(C["paddleH"]) / 2.0, 0.0, float(C["height"] - C["paddleH"]))
    err = target - float(agent_y)
    return clip(err, -speed, speed)


def right_dy(
    opponent: str,
    state: dict[str, Any],
    C: dict[str, Any],
    fly: ApproachCommitController | None,
) -> tuple[float, bool]:
    if opponent != "approach" or fly is None:
        return float(lag_opponent_dy(state, C)), False
    cmd = fly.step_command(mirror_right_state(state, C))
    return float(cmd["dy"]), bool(cmd["commit"])


def live_label(opponent: str) -> dict[str, str]:
    if opponent == "approach":
        return {
            "caption": "human vs approach_commit",
            "left_name": "YOU",
            "right_name": "GOALIE",
            "print": (
                "live controller: human vs approach_commit\n"
                "left: W/S or arrows or mouse\n"
                "right: approach_commit (centering + 24-frame occupy-the-Y)\n"
                "this court is a toy, not a fly title"
            ),
        }
    return {
        "caption": "human vs lag_chase",
        "left_name": "YOU",
        "right_name": "LAG",
        "print": (
            "live controller: human vs lag_chase\n"
            "left: W/S or arrows or mouse\n"
            "right: env lag paddle (0.75x speed)\n"
            "this court is a toy, not a fly title"
        ),
    }


def play_headless(
    *,
    opponent: str = "approach",
    frames: int = 120,
    seed: int = 0,
    oracle_y: bool = False,
    agent_action: int = 0,
) -> dict[str, Any]:
    """Drive a few frames without a window. Tests the opponent slot."""
    C = load_constants()
    env = FlyPongEnv(render_mode=None)
    fly = None
    if opponent == "approach":
        fly = ApproachCommitController(oracle_y=oracle_y)
    env.reset(seed=seed)
    if fly is not None:
        fly.reset()
    commits = 0
    try:
        for _ in range(int(frames)):
            raw = env.unwrapped._state
            dy, commit = right_dy(opponent, raw, C, fly)
            if commit:
                commits += 1
            env.step(agent_action, opp_dy=dy)
            if env.unwrapped._state["terminated"]:
                break
        raw = env.unwrapped._state
        return {
            "agent_y": float(raw["agent_y"]),
            "opp_y": float(raw["opp_y"]),
            "agent_score": int(raw["agent_score"]),
            "opp_score": int(raw["opp_score"]),
            "commits": commits,
            "opponent": opponent,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opponent",
        choices=("approach", "lag"),
        default="approach",
        help="Right paddle. approach_commit is the 24-frame goalie.",
    )
    parser.add_argument("--oracle_y", action="store_true", help="True ball Y for approach_commit.")
    args = parser.parse_args()
    names = live_label(args.opponent)
    print(names["print"], flush=True)

    import os

    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame = __import__("pygame")
    C = load_constants()
    env = FlyPongEnv(render_mode=None)
    fly = ApproachCommitController(oracle_y=bool(args.oracle_y)) if args.opponent == "approach" else None
    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((int(C["width"]), int(C["height"])))
    pygame.display.set_caption(names["caption"])
    clock = pygame.time.Clock()
    env.reset()
    if fly is not None:
        fly.reset()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            keys = pygame.key.get_pressed()
            raw = env.unwrapped._state
            agent_dy = human_dy(keys, pygame.mouse.get_pos()[1], float(raw["agent_y"]), C, pygame)
            opp, commit = right_dy(args.opponent, raw, C, fly)
            env.step(0, opp_dy=opp, agent_dy=agent_dy)
            hud = {
                "caption": names["caption"],
                "left_name": names["left_name"],
                "right_name": names["right_name"],
                "commit": commit,
            }
            draw_court(screen, env.unwrapped._state, C, hud)
            pygame.display.flip()
            clock.tick(60)
            if env.unwrapped._state["terminated"]:
                env.reset()
                if fly is not None:
                    fly.reset()
    finally:
        env.close()
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
