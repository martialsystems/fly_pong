"""Keyboard sanity-check of the Gym physics."""

from __future__ import annotations

from fly_pong.env import FlyPongEnv


def main() -> None:
    pygame = __import__("pygame")
    env = FlyPongEnv(render_mode="human")
    obs, info = env.reset()
    running = True
    try:
        while running:
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                action = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                action = 2
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
