import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tensors():
    g = torch.Generator().manual_seed(7)
    head = torch.randn(41, 12, generator=g) * 0.3
    hidden = torch.randn(6, 12, generator=g)
    teacher = torch.randn(6, 41, generator=g).softmax(-1)
    return head, hidden, teacher


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires a CUDA device and Triton")
