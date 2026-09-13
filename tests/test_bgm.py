from __future__ import annotations

import wave

import numpy as np

from fly_pong.music import BGM, HIT, play_hit, start_music


def test_bgm_wav_is_a_loop_not_silence():
    assert BGM.is_file()
    with wave.open(str(BGM), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 22050
        n = w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16)
    dur = n / 22050.0
    assert 8.0 < dur < 14.0
    rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))
    assert rms > 200.0


def test_hit_wav_is_a_short_click():
    assert HIT.is_file()
    with wave.open(str(HIT), "rb") as w:
        assert w.getframerate() == 22050
        n = w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16)
    assert 0.04 < n / 22050.0 < 0.15
    assert float(np.max(np.abs(raw))) > 1000


def test_start_music_survives_missing_mixer():
    class Mixer:
        @staticmethod
        def get_init():
            raise RuntimeError("no mixer")

    class Boom:
        mixer = Mixer()

    assert start_music(Boom()) is False
    assert play_hit(Boom()) is False
