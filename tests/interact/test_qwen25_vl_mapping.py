from types import SimpleNamespace

import torch

from slime_plugins.models.qwen25_vl import export_tensor, hf_tensor


def test_qkv_load_export_round_trip():
    config = SimpleNamespace(num_key_value_heads=2, num_attention_heads=4, hidden_size=8)
    args = SimpleNamespace(num_query_groups=2, num_attention_heads=4, hidden_size=8, kv_channels=2)
    for suffix, trailing in (("weight", (8,)), ("bias", ())):
        weights = {f"model.layers.0.self_attn.{p}_proj.{suffix}": torch.randn(rows, *trailing)
                   for p, rows in (("q", 8), ("k", 4), ("v", 4))}
        class Reader(dict):
            get_tensor = dict.__getitem__
        reader = Reader(weights)
        name = "module.module.language_model.decoder.layers.0.self_attention.linear_qkv." + suffix
        packed = hf_tensor(name, reader, config)
        exported = dict(export_tensor(args, name, packed))
        assert exported.keys() == weights.keys()
        for key in weights:
            assert torch.equal(exported[key], weights[key])


def test_visual_passthrough_and_language_vocab_unpadding():
    weight = torch.randn(12, 8)
    args = SimpleNamespace(vocab_size=9)
    name, actual = export_tensor(args, "module.module.language_model.embedding.word_embeddings.weight", weight)[0]
    assert name == "model.embed_tokens.weight" and torch.equal(actual, weight[:9])
    visual = "visual.blocks.0.attn.qkv.weight"
    assert export_tensor(args, "module.module." + visual, weight)[0][0] == visual
    loaded = hf_tensor("module.module." + visual, SimpleNamespace(get_tensor=lambda name: weight), None)
    assert loaded is weight
