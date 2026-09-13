#!/usr/bin/env python3
"""Synthesize a looping fly-court bed: wing buzz under a D-dorian ostinato."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fly_pong" / "assets" / "bgm.wav"
HIT = ROOT / "fly_pong" / "assets" / "hit.wav"
SR = 22050
BPM = 96
BARS = 4
BEATS = BARS * 4


def _env_pluck(t: np.ndarray, decay: float = 7.5) -> np.ndarray:
    e = np.exp(-np.maximum(t, 0.0) * decay)
    e *= (t >= 0.0) & (t < 2.0)
    return e


def _lute(n: int, freq: float, amp: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    sig = np.zeros(n, dtype=np.float64)
    for k, a in enumerate((1.00, 0.42, 0.18, 0.08, 0.03), start=1):
        sig += a * np.sin(2.0 * np.pi * freq * k * t)
    return amp * _env_pluck(t) * sig


def _tone(n: int, freq: float, amp: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


def main() -> None:
    beat = 60.0 / BPM
    n = int(round(BEATS * beat * SR))
    t = np.arange(n, dtype=np.float64) / SR
    rng = np.random.default_rng(7)

    # Quiet fifth drone (D2 + A2).
    drone = _tone(n, 73.42, 0.07) + _tone(n, 110.00, 0.05)
    drone *= 0.85 + 0.15 * np.sin(2.0 * np.pi * 0.125 * t)

    # Housefly wingbed: ~196 Hz AM on band-limited noise.
    noise = rng.normal(0.0, 1.0, n)
    lp = np.empty(n, dtype=np.float64)
    acc = 0.0
    for i, x in enumerate(noise):
        acc = 0.14 * x + 0.86 * acc
        lp[i] = acc
    wing = 0.55 + 0.45 * np.sin(2.0 * np.pi * 196.0 * t)
    wander = 0.75 + 0.25 * np.sin(2.0 * np.pi * 0.35 * t)
    buzz = 0.09 * (noise - lp) * wing * wander

    # D dorian ostinato, one note per beat.
    dorian = {
        "D3": 146.83,
        "E3": 164.81,
        "F3": 174.61,
        "G3": 196.00,
        "A3": 220.00,
        "C4": 261.63,
        "D4": 293.66,
        "A4": 440.00,
    }
    pattern = (
        "D4",
        "A3",
        "G3",
        "F3",
        "E3",
        "D3",
        "A3",
        "G3",
        "D4",
        "F3",
        "A3",
        "G3",
        "E3",
        "C4",
        "A3",
        "D4",
    )
    melody = np.zeros(n, dtype=np.float64)
    beat_n = int(round(beat * SR))
    for i, name in enumerate(pattern):
        start = i * beat_n
        chunk = _lute(min(beat_n * 2, n - start), dorian[name], 0.22)
        melody[start : start + chunk.size] += chunk

    mix = drone + buzz + melody
    # Loop crossfade.
    fade = int(0.18 * SR)
    ramp = np.linspace(0.0, 1.0, fade)
    mix[-fade:] *= 1.0 - ramp
    mix[-fade:] += mix[:fade] * ramp
    mix[:fade] = mix[-fade:]

    peak = float(np.max(np.abs(mix))) or 1.0
    pcm = np.int16(np.clip(mix / peak * 0.72, -1.0, 1.0) * 32767)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {OUT.relative_to(ROOT)}  {n / SR:.2f}s  peak={peak:.3f}")
    _write_hit()


def _write_hit() -> None:
    n = int(0.09 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    rng = np.random.default_rng(3)
    env = np.exp(-t * 58.0)
    thump = np.sin(2.0 * np.pi * 148.0 * t) * 0.65
    thump += np.sin(2.0 * np.pi * 296.0 * t) * 0.22
    click = rng.normal(0.0, 1.0, n) * 0.28
    mix = (thump + click) * env
    peak = float(np.max(np.abs(mix))) or 1.0
    pcm = np.int16(np.clip(mix / peak * 0.85, -1.0, 1.0) * 32767)
    with wave.open(str(HIT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {HIT.relative_to(ROOT)}  {n / SR:.3f}s")


if __name__ == "__main__":
    main()
