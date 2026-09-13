from __future__ import annotations

from fly_pong.court import button_rects, hit
from fly_pong.scores import load, record, save


def test_record_sorts_wins_then_human_points(tmp_path):
    path = tmp_path / "scores.json"
    record(2, 11, path=path)
    record(11, 4, path=path)
    record(11, 9, path=path)
    rows = load(path)
    assert rows[0]["win"] is True
    assert rows[0]["human"] == 11
    assert rows[0]["fly"] == 4
    assert rows[-1]["win"] is False


def test_save_roundtrip(tmp_path):
    path = tmp_path / "scores.json"
    save([{"human": 7, "fly": 11, "win": False}], path)
    rows = load(path)
    assert rows == [{"human": 7, "fly": 11, "win": False}]


def test_menu_hit_boxes():
    rects = button_rects(1200, 720)
    start = rects["start"]
    cx = start[0] + start[2] // 2
    cy = start[1] + start[3] // 2
    assert hit(start, (cx, cy)) is True
    assert hit(start, (0, 0)) is False
    assert "scores" in rects and "quit" in rects and "back" in rects
