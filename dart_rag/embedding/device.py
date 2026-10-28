"""연산 장치 판별. torch 가 없으면 cpu (hashing/openai 는 torch 불필요)."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def resolve_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_batch_size(device: str, requested: int | None) -> int:
    if requested:
        return int(requested)
    return 64 if device == "cuda" else 16


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"name": p.name, "vram_gb": round(p.total_memory / 1024 ** 3, 1)}
    except ImportError:
        pass
    return {}
