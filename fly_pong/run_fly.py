"""Closed loop: Gym body, T4/T5 motion decode. Lab only."""

from __future__ import annotations

from fly_pong.bridge import FlyBrainBridge
from fly_pong.env import FlyPongEnv


def main() -> None:
    env = FlyPongEnv(render_mode="human")
    brain = FlyBrainBridge(n_ommatidia=32)
    obs, info = env.reset()
    brain.reset()
    try:
        while True:
            action, neural = brain.brain_step(info["state"])
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                state = info["state"]
                print(
                    f"score {state['agent_score']}-{state['opp_score']}  "
                    f"steer={neural['steering']:.4f}  action={action}"
                )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
