"""Hook for a later Neurokernel session. FlyBrainLab is not a pip-in Pong brain."""

from __future__ import annotations

from typing import Any


class FlyBrainLabAdapter:
    """Swap this in when an FBL/Neurokernel server is running."""

    def __init__(self, client: Any = None):
        self.client = client

    def inject_stimulus(self, voltage_vector: Any) -> None:
        raise NotImplementedError(
            "Requires a local Neurokernel session. Public FFBO backends do not execute circuits."
        )

    def read_motor(self) -> int:
        raise NotImplementedError(
            "Requires a local Neurokernel session to decode DNs / steering motoneurons."
        )
