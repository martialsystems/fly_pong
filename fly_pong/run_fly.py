"""Closed loop: Gym body, T4/T5 motion decode. Lab only."""

from __future__ import annotations

from fly_pong.bridge import FlyBrainBridge
from fly_pong.env import FlyPongEnv
from fly_pong.physics import dy_from_action


def main() -> None:
    env = FlyPongEnv(render_mode="human")
    brain = FlyBrainBridge(n_ommatidia=32)
    obs, info = env.reset()
    brain.reset()
    try:
        while True:
            action, neural = brain.brain_step(info["state"])
            dy = brain.motor.push(dy_from_action(int(action), C=env.unwrapped.C))
            delayed = 0
            if dy < -0.5:
                delayed = 1
            elif dy > 0.5:
                delayed = 2
            obs, reward, terminated, truncated, info = env.step(delayed)
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
