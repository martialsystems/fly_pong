"""Commit-if-reachable veto. Contact, not placement."""

from __future__ import annotations


def reach_frames(y_pred: float, paddle_center: float, paddle_speed: float) -> float:
    speed = max(float(paddle_speed), 1e-6)
    return abs(float(y_pred) - float(paddle_center)) / speed


def should_commit(
    *,
    incoming: bool,
    tau: float,
    y_pred: float,
    paddle_center: float,
    paddle_speed: float,
    window: float,
) -> bool:
    """Lunge only if the intercept is in range before contact.

    tau: frames to own paddle. window: short time-to-contact (AIM_N).
    Firing when reach > tau is the 7/40 setpoint path. Do not.
    """
    if not incoming:
        return False
    if float(tau) > float(window):
        return False
    return reach_frames(y_pred, paddle_center, paddle_speed) <= float(tau)
