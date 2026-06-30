"""
Centralized device resolution for HybridMind.

All model loads (embedding, reranker, ColBERT, GNN) should call resolve_device()
rather than duplicating CUDA detection logic.
"""
import logging
import os

logger = logging.getLogger(__name__)


def resolve_device(pref: str = "auto") -> str:
    """
    Resolve a device preference to a concrete device string.

    pref values:
    - "auto"  : cuda > mps > cpu (default)
    - "cuda"  : force CUDA; raises RuntimeError if unavailable
    - "mps"   : force MPS (Apple Silicon); raises RuntimeError if unavailable
    - "cpu"   : force CPU

    Returns one of: "cuda", "mps", "cpu"
    """
    pref = (pref or "auto").strip().lower()

    try:
        import torch
        _has_torch = True
    except ImportError:
        _has_torch = False

    if pref == "cpu":
        return "cpu"

    if pref == "cuda":
        if not _has_torch:
            raise RuntimeError("HYBRIDMIND_DEVICE=cuda but torch is not installed")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "HYBRIDMIND_DEVICE=cuda but CUDA is not available on this machine. "
                "Set HYBRIDMIND_DEVICE=auto or install CUDA-enabled torch."
            )
        return "cuda"

    if pref == "mps":
        if not _has_torch:
            raise RuntimeError("HYBRIDMIND_DEVICE=mps but torch is not installed")
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                "HYBRIDMIND_DEVICE=mps but MPS is not available on this machine."
            )
        return "mps"

    # auto: cuda > mps > cpu
    if _has_torch:
        if torch.cuda.is_available():
            logger.info("device.resolve_device: auto → cuda")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("device.resolve_device: auto → mps")
            return "mps"

    logger.info("device.resolve_device: auto → cpu")
    return "cpu"


def gpu_info() -> dict:
    """Return a status dict for the /health endpoint."""
    try:
        import torch
    except ImportError:
        return {"status": "healthy", "device": "cpu", "device_name": None, "cuda_version": None}

    if torch.cuda.is_available():
        return {
            "status": "healthy",
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        }

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {
            "status": "healthy",
            "device": "mps",
            "device_name": "Apple MPS",
            "cuda_version": None,
        }

    return {"status": "healthy", "device": "cpu", "device_name": None, "cuda_version": None}
