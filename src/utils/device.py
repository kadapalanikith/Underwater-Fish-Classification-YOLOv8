"""
Hardware Device Selector and Diagnostic Utility
Handles modern architectures (e.g. Blackwell sm_120 / RTX 50-series)
by safely validating kernel execution before dispatching to GPU or falling back to CPU.
"""

import logging
import torch

logger = logging.getLogger("DeepSea-Device")

_SELECTED_DEVICE = None


def get_optimal_device() -> str:
    global _SELECTED_DEVICE
    if _SELECTED_DEVICE is not None:
        return _SELECTED_DEVICE

    if not torch.cuda.is_available():
        _SELECTED_DEVICE = "cpu"
        logger.info("CUDA not available. Using CPU.")
        return _SELECTED_DEVICE

    # Test actual kernel execution to catch compute capability mismatches (e.g. sm_120 on cu126)
    try:
        test_tensor = torch.zeros((1, 1), device="cuda:0")
        _ = test_tensor + 1.0
        torch.cuda.synchronize()
        _SELECTED_DEVICE = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name} ({_SELECTED_DEVICE})")
    except Exception as e:
        logger.warning(
            f"GPU detected ({torch.cuda.get_device_name(0)}), but CUDA kernel execution failed ({e}). "
            "Falling back safely to CPU execution."
        )
        _SELECTED_DEVICE = "cpu"

    return _SELECTED_DEVICE
