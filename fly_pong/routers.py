"""Softmax gates over named channels."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxRouter(nn.Module):
    def __init__(self, n: int, bias: np.ndarray | None = None):
        super().__init__()
        init = torch.zeros(int(n))
        if bias is not None:
            init = torch.as_tensor(np.asarray(bias, dtype=np.float32))
        self.logits = nn.Parameter(init)

    def gates(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)

    def gates_np(self) -> np.ndarray:
        with torch.no_grad():
            return self.gates().detach().cpu().numpy().astype(np.float32)

    def command(self, feats: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(feats, dtype=np.float32), dtype=torch.float32)
        return torch.dot(self.gates(), x)

    def command_np(self, feats: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.command(feats).item())
