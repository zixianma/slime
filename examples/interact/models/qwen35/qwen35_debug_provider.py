"""Diagnostic-only anomaly tracing around the official Qwen3.5-VL provider."""
import torch
from slime_plugins.models.qwen3_5_vl import get_qwen3_5_vl_model_provider


def get_model_provider(args, config, vp_stage):
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    return get_qwen3_5_vl_model_provider(args, config, vp_stage)
