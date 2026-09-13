"""Human left paddle vs lag or approach_commit. Python court only."""

from __future__ import annotations

import argparse
from typing import Any

from fly_pong.commit import ApproachCommitController
from fly_pong.constants import load_constants
from fly_pong.court import button_rects, court_from_window_y
from fly_pong.court import draw as draw_court
from fly_pong.court import draw_scores, draw_title, frame_size, hit, play_button_rects
from fly_pong.env import FlyPongEnv
from fly_pong.music import play_hit, start_music, stop_music
from fly_pong.physics import clip, lag_opponent_dy, mirror_right_state, paddle_contact
from fly_pong.scores import load as load_scores
from fly_pong.scores import record as record_score

# Playable court only. Locked evals keep bounceGain 1.02 and no cap.
PLAY_BOUNCE_GAIN = 1.008
PLAY_BALL_SPEED_MAX = 8.5


def human_dy(keys, mouse_y: float, agent_y: float, C: dict[str, Any], pygame) -> float:
    speed = float(C["paddleSpeed"])
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return -speed
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return speed
    court_y = court_from_window_y(mouse_y)
    target = clip(court_y - float(C["paddleH"]) / 2.0, 0.0, float(C["height"] - C["paddleH"]))
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
            "caption": "human vs fly",
            "left_name": "HUMAN",
            "right_name": "FLY",
            "print": (
                "live controller: human vs fly\n"
                "left: W/S or arrows or mouse\n"
                "right: approach_commit (centering + 24-frame occupy-the-Y)"
            ),
        }
    return {
        "caption": "human vs lag",
        "left_name": "HUMAN",
        "right_name": "LAG",
        "print": (
            "live controller: human vs lag\n"
            "left: W/S or arrows or mouse\n"
            "right: env lag paddle (0.75x speed)"
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


def _boot_match(env: FlyPongEnv, fly: ApproachCommitController | None) -> None:
    env.reset()
    if fly is not None:
        fly.reset()


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
    print("menu: Start Game, High Scores, Reset (R). M mutes the bed.", flush=True)

    pygame = __import__("pygame")
    C = load_constants()
    env = FlyPongEnv(render_mode=None)
    env.C["bounceGain"] = PLAY_BOUNCE_GAIN
    env.C["ballSpeedMax"] = PLAY_BALL_SPEED_MAX
    fly = ApproachCommitController(oracle_y=bool(args.oracle_y)) if args.opponent == "approach" else None
    try:
        pygame.mixer.pre_init(22050, size=-16, channels=1, buffer=512)
    except Exception:
        pass
    pygame.init()
    pygame.display.init()
    start_music(pygame)
    fw, fh = frame_size(C)
    screen = pygame.display.set_mode((fw, fh))
    pygame.display.set_caption(names["caption"])
    clock = pygame.time.Clock()
    mode = "title"
    recorded = False
    _boot_match(env, fly)
    running = True
    try:
        while running:
            mouse = pygame.mouse.get_pos()
            click = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    click = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if mode == "play":
                            mode = "title"
                        elif mode == "scores":
                            mode = "title"
                        else:
                            running = False
                    elif event.key == pygame.K_RETURN and mode == "title":
                        _boot_match(env, fly)
                        recorded = False
                        mode = "play"
                    elif event.key in (pygame.K_r,) and mode == "play":
                        _boot_match(env, fly)
                        recorded = False
                    elif event.key == pygame.K_m:
                        try:
                            vol = pygame.mixer.music.get_volume()
                            pygame.mixer.music.set_volume(0.0 if vol > 0.05 else 0.38)
                        except Exception:
                            pass
            keys = pygame.key.get_pressed()
            menus = button_rects(fw, fh)
            plays = play_button_rects(fw, fh)

            if mode == "title":
                if click and hit(menus["start"], mouse):
                    _boot_match(env, fly)
                    recorded = False
                    mode = "play"
                elif click and hit(menus["scores"], mouse):
                    mode = "scores"
                elif click and hit(menus["quit"], mouse):
                    running = False
                draw_title(screen, C, mouse=mouse)
            elif mode == "scores":
                if click and hit(menus["back"], mouse):
                    mode = "title"
                draw_scores(screen, C, load_scores(), mouse=mouse)
            else:
                commit = False
                if click and hit(plays["reset"], mouse):
                    _boot_match(env, fly)
                    recorded = False
                elif click and hit(plays["menu"], mouse):
                    mode = "title"
                else:
                    raw = env.unwrapped._state
                    if not raw["terminated"]:
                        prev = dict(raw)
                        agent_dy = human_dy(keys, mouse[1], float(raw["agent_y"]), C, pygame)
                        opp, commit = right_dy(args.opponent, raw, C, fly)
                        env.step(0, opp_dy=opp, agent_dy=agent_dy)
                        if paddle_contact(prev, env.unwrapped._state):
                            play_hit(pygame)
                    elif not recorded:
                        record_score(int(raw["agent_score"]), int(raw["opp_score"]))
                        recorded = True
                if mode == "play":
                    hud = {
                        "caption": names["caption"],
                        "left_name": names["left_name"],
                        "right_name": names["right_name"],
                        "commit": commit,
                        "mouse": mouse,
                        "show_play_buttons": True,
                    }
                    draw_court(screen, env.unwrapped._state, C, hud)

            pygame.display.flip()
            clock.tick(60)
    finally:
        stop_music(pygame)
        env.close()
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
