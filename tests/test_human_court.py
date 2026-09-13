from __future__ import annotations

from fly_pong.commit import ApproachCommitController
from fly_pong.constants import load_constants
from fly_pong.env import FlyPongEnv
from fly_pong.physics import lag_opponent_dy, mirror_right_state
from fly_pong.run_human import live_label, play_headless, right_dy


def test_opp_dy_moves_right_paddle():
    env = FlyPongEnv(render_mode=None)
    env.reset(seed=0)
    y0 = float(env.unwrapped._state["opp_y"])
    env.step(0, opp_dy=7.0)
    y1 = float(env.unwrapped._state["opp_y"])
    env.close()
    assert y1 > y0


def test_default_step_still_uses_lag():
    env = FlyPongEnv(render_mode=None)
    clone = FlyPongEnv(render_mode=None)
    env.reset(seed=1)
    clone.reset(seed=1)
    dy = lag_opponent_dy(env.unwrapped._state, env.C)
    env.step(0)
    clone.step(0, opp_dy=dy)
    assert env.unwrapped._state["opp_y"] == clone.unwrapped._state["opp_y"]
    env.close()
    clone.close()


def test_mirror_right_flips_x_and_vx():
    C = load_constants()
    raw = {
        "ball_x": 100.0,
        "ball_y": 180.0,
        "ball_vx": 5.0,
        "ball_vy": -1.0,
        "opp_y": 40.0,
        "agent_y": 200.0,
        "agent_score": 1,
        "opp_score": 2,
    }
    m = mirror_right_state(raw, C)
    assert abs(m["ball_x"] - (float(C["width"]) - 100.0)) < 1e-9
    assert m["ball_vx"] == -5.0
    assert m["paddle_y"] == 40.0
    assert m["opp_y"] == 200.0


def test_approach_right_paddle_tracks_incoming():
    C = load_constants()
    fly = ApproachCommitController(oracle_y=True)
    fly.reset()
    # Ball heading at the right paddle, below it.
    state = {
        "ball_x": float(C["oppX"]) - 40.0,
        "ball_y": 280.0,
        "ball_vx": 5.0,
        "ball_vy": 0.0,
        "agent_y": 150.0,
        "opp_y": 40.0,
        "agent_score": 0,
        "opp_score": 0,
    }
    dy, _commit = right_dy("approach", state, C, fly)
    assert dy > 0.0


def test_play_headless_approach_differs_from_lag():
    lag = play_headless(opponent="lag", frames=90, seed=2)
    app = play_headless(opponent="approach", frames=90, seed=2)
    assert lag["opponent"] == "lag"
    assert app["opponent"] == "approach"
    assert lag["opp_y"] != app["opp_y"]


def test_live_label_is_not_a_title():
    a = live_label("approach")
    assert a["caption"] == "human vs approach_commit"
    assert "not a fly title" in a["print"]
    assert "approach_commit" in a["print"]


def test_medieval_rgb_is_not_the_old_void():
    env = FlyPongEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    env.close()
    assert frame is not None
    assert frame.shape == (360, 480, 3)
    corner = tuple(int(v) for v in frame[4, 4])
    assert corner != (12, 12, 18)
    torch = tuple(int(v) for v in frame[9, 36])
    assert torch[0] > 150 and torch[1] > 100  # gold finial, not the old yellow ball
