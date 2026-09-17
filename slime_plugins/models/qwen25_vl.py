"""Opt-in Qwen2.5-VL integration for the official Slime checkout.

Uses NVIDIA Megatron Bridge's model provider, and registers only this model's
HF load/export mappings. No OpenWebRL code or training-loop replacement.
Initial validation profile: one sample per microbatch, CP1/PP1, frozen vision.
"""
_registered = False
_provider = None


def hf_tensor(name, reader, config):
    from slime.backends.megatron_utils.hf_to_megatron.common import strip_mcore_wrappers
    from slime.backends.megatron_utils.hf_to_megatron.qwen import qwen_hf_tensor
    clean = strip_mcore_wrappers(name)
    if clean.startswith("visual."):
        return reader.get_tensor(clean)
    return qwen_hf_tensor(name, reader, config)


def export_tensor(args, name, param):
    from slime.backends.megatron_utils.hf_to_megatron.common import strip_mcore_wrappers
    from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf
    clean = strip_mcore_wrappers(name)
    if clean.startswith("visual."):
        return [(clean, param)]
    if clean in ("embedding.word_embeddings.weight", "output_layer.weight"):
        param = param[:args.vocab_size]
    return convert_qwen2_to_hf(args, "module.module." + clean, param)


def register(args=None):
    global _registered
    if _registered:
        return
    import slime.backends.megatron_utils.hf_to_megatron as loading
    import slime.backends.megatron_utils.megatron_to_hf as export
    loading._LOADERS["qwen2_5_vl"] = hf_tensor
    original = export._convert_to_hf_core

    def convert(args, model_name, name, param):
        family = model_name.lower().replace("_", "").replace("-", "").replace(".", "")
        if "qwen25vl" in family:
            return export_tensor(args, name, param)
        return original(args, model_name, name, param)
    export._convert_to_hf_core = convert
    _registered = True


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    global _provider
    from megatron.training.global_vars import get_args
    from megatron.bridge import AutoBridge
    args = get_args()
    if args.context_parallel_size != 1 or args.pipeline_model_parallel_size != 1 or args.micro_batch_size != 1:
        raise ValueError("initial Qwen2.5-VL bridge requires CP1, PP1, microbatch1")
    if getattr(args, "use_dynamic_batch_size", False):
        raise ValueError("multi-sample packing is not validated by this provider")
    register()
    if _provider is None:
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        _provider = bridge.to_megatron_provider(load_weights=False)
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "sequence_parallel",
                     "context_parallel_size", "expert_model_parallel_size", "expert_tensor_parallel_size"):
            setattr(_provider, name, getattr(args, name))
        _provider.vision_config._attn_implementation = "sdpa"
        _provider.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
        _provider.freeze_vision_model = True
        _provider.freeze_vision_projection = True
        _provider.seq_length = args.rollout_max_context_len
        _provider.variable_seq_lengths = True
        _provider.moe_token_dispatcher_type = "alltoall"
        _provider.attention_dropout = 0.0
        _provider.hidden_dropout = 0.0
        _provider.finalize()
    return _provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
