#!/usr/bin/env python3
"""Synthesize a perfectly looping fly-court bed: wrapping D-dorian arp plus wing buzz."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fly_pong" / "assets" / "bgm.wav"
HIT = ROOT / "fly_pong" / "assets" / "hit.wav"
SR = 22050
# 11025 samples/beat at 22050 Hz is exactly 0.5 s → 120 BPM. 20 beats = 10.0 s.
BEAT_N = 11025
BEATS = 20
N = BEAT_N * BEATS  # 220500
LOOP_HZ = SR / float(N)  # 0.1 Hz. Every oscillator is k * LOOP_HZ.


def _harm(freq: float) -> float:
    """Snap to an integer number of cycles in the loop."""
    return round(freq / LOOP_HZ) * LOOP_HZ


def _tone(freq: float, amp: float) -> np.ndarray:
    t = np.arange(N, dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * _harm(freq) * t)


def _lute(n: int, freq: float, amp: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    f = _harm(freq)
    sig = np.zeros(n, dtype=np.float64)
    for k, a in enumerate((1.00, 0.40, 0.16, 0.07, 0.03), start=1):
        sig += a * np.sin(2.0 * np.pi * f * k * t)
    env = np.exp(-t * 9.0)
    return amp * env * sig


def _add_wrapped(buf: np.ndarray, start: int, chunk: np.ndarray) -> None:
    n = buf.size
    i = int(start) % n
    left = n - i
    if chunk.size <= left:
        buf[i : i + chunk.size] += chunk
        return
    buf[i:] += chunk[:left]
    rest = chunk[left:]
    buf[: rest.size] += rest


def _arp_steps(n_steps: int) -> list[float]:
    """Up-down arpeggiator on D-dorian tetrad D-F-A-C. 8-step cycle divides 40 eighths."""
    chord = (146.83, 174.61, 220.00, 261.63)  # D3 F3 A3 C4
    cycle = chord + tuple(reversed(chord))
    return [cycle[i % len(cycle)] for i in range(n_steps)]


def main() -> None:
    t = np.arange(N, dtype=np.float64) / SR

    drone = _tone(73.42, 0.07) + _tone(110.00, 0.05)
    drone *= 0.88 + 0.12 * np.sin(2.0 * np.pi * LOOP_HZ * t)

    wing_f = _harm(196.0)
    wing = 0.55 + 0.45 * np.sin(2.0 * np.pi * wing_f * t)
    wander = 0.78 + 0.22 * np.sin(2.0 * np.pi * (4 * LOOP_HZ) * t)
    buzz = 0.08 * np.sin(2.0 * np.pi * wing_f * t)
    buzz += 0.03 * np.sin(2.0 * np.pi * (2 * wing_f) * t)
    buzz *= wing * wander

    # Eighth-note arp: 2 steps per beat, 40 steps, tails wrap into bar 1.
    steps = _arp_steps(BEATS * 2)
    step_n = BEAT_N // 2
    arp = np.zeros(N, dtype=np.float64)
    tail = step_n * 3
    for i, freq in enumerate(steps):
        chunk = _lute(tail, freq, 0.18)
        _add_wrapped(arp, i * step_n, chunk)

    mix = drone + buzz + arp
    peak = float(np.max(np.abs(mix))) or 1.0
    pcm = np.int16(np.clip(mix / peak * 0.72, -1.0, 1.0) * 32767)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {OUT.relative_to(ROOT)}  {N / SR:.2f}s  peak={peak:.3f}")
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
