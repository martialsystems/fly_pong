from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from fly_pong.court import frame_size
from fly_pong.env import FlyPongEnv
from fly_pong.obs import OBS_HIGH, OBS_LOW, encode, encode_mirrored_right


def test_gymnasium_api():
    env = FlyPongEnv(render_mode=None)
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_reset_obs_in_space():
    env = FlyPongEnv(render_mode=None)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "state" in info
    assert set(info["state"]) >= {"ball_x", "ball_y", "paddle_y", "opp_y"}
    env.close()


def test_step_five_tuple_and_seed():
    env_a = FlyPongEnv(render_mode=None)
    env_b = FlyPongEnv(render_mode=None)
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    np.testing.assert_allclose(obs_a, obs_b)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env_a.step(env_a.action_space.sample())
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert truncated is False
        assert "state" in info
        if terminated:
            obs, info = env_a.reset()
            break
    env_a.close()
    env_b.close()


def test_rgb_array_shape():
    env = FlyPongEnv(render_mode="rgb_array")
    env.reset(seed=1)
    frame = env.render()
    assert frame is not None
    fw, fh = frame_size(env.C)
    assert frame.shape == (fh, fw, 3)
    env.close()


def test_mirror_swaps_paddles_and_flips_x():
    env = FlyPongEnv(render_mode=None)
    obs, info = env.reset(seed=3)
    state = env.unwrapped._state
    left = encode(state)
    right = encode_mirrored_right(state)
    np.testing.assert_allclose(left, obs)
    assert np.isclose(left[0] + right[0], 1.0)
    assert np.isclose(left[2] + right[2], 0.0, atol=1e-6)
    assert np.isclose(right[4], left[5])
    assert np.isclose(right[5], left[4])
    env.close()
    assert OBS_LOW.shape == OBS_HIGH.shape == (6,)
