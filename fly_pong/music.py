"""Loop the court bed. Fail open if the mixer cannot start."""

from __future__ import annotations

from pathlib import Path

BGM = Path(__file__).resolve().parent / "assets" / "bgm.wav"
HIT = Path(__file__).resolve().parent / "assets" / "hit.wav"
_hit = None


def start_music(pygame, *, volume: float = 0.38) -> bool:
    path = BGM
    if not path.is_file():
        return False
    try:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.set_volume(float(volume))
        pygame.mixer.music.play(-1)
        return True
    except Exception:
        return False


def stop_music(pygame) -> None:
    try:
        if pygame.mixer.get_init() is not None:
            pygame.mixer.music.stop()
    except Exception:
        return


def play_hit(pygame) -> bool:
    global _hit
    if not HIT.is_file():
        return False
    try:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        if _hit is None:
            _hit = pygame.mixer.Sound(str(HIT))
            _hit.set_volume(0.62)
        _hit.play()
        return True
    except Exception:
        return False
