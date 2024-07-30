# adapted from https://github.com/InternLM/xtuner/blob/main/xtuner/model/modules/dispatch/__init__.py

import types

from packaging.version import Version

import transformers
from internlm.model.modules.dispatch.utils import LazyObject

TRANSFORMERS_VERSION = Version(transformers.__version__)

LOWEST_TRANSFORMERS_VERSION = dict(
    InternLMForCausalLM=Version("4.36"),
)

ATTN_DISPATCH_MAPPING = dict(
    InternLMAttention=LazyObject("internlm.model.modules.dispatch.internlm", "internlm_attn_forward"),
    InternLM2Attention=LazyObject("internlm.model.modules.dispatch.internlm2", "internlm2_attn_forward"),
)

VARLEN_ATTN_DISPATCH_MAPPING = dict(
    InternLMAttention=LazyObject("internlm.model.modules.dispatch.internlm", "internlm_varlen_attn_forward"),
    InternLM2Attention=LazyObject("internlm.model.modules.dispatch.internlm2", "internlm2_varlen_attn_forward"),
)

EMBED_REPLACE_MAPPING = dict(
    Embedding=LazyObject("internlm.model.modules.embedding", "Embedding1D"),
)

NORM_REPLACE_MAPPING = dict(
    InternLMRMSNorm=LazyObject("internlm.model.modules.norm", "new_layer_norm"),
    InternLM2RMSNorm=LazyObject("internlm.model.modules.norm", "new_layer_norm"),
)

LINEAR_REPLACE_MAPPING = dict(
    Linear=LazyObject("internlm.model.modules.linear", "new_linear"),
)

NORM2NEW_NORM_NAME_MAPPING = dict(
    input_layernorm="rmsnorm",
    post_attention_layernorm="rmsnorm",
    norm="rmsnorm",
    attention_norm="rmsnorm",
    ffn_norm="rmsnorm",
)

LINEAR2NEW_LINEAR_NAME_MAPPING = dict(
    q_proj="wq",
    k_proj="wk",
    v_proj="wv",
    o_proj="wo",
    gate_proj="w1",
    down_proj="w2",
    up_proj="w3",
    lm_head="head",
)


# hack: dispatch forward for attn
def dispatch_attn_forward(model):
    attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in ATTN_DISPATCH_MAPPING:
            if attn_forward is None:
                attn_forward = ATTN_DISPATCH_MAPPING[name]
                attn_forward = attn_forward.build()
            module.forward = types.MethodType(attn_forward, module)


# hack: dispatch forward for varlen attn
def dispatch_varlen_attn_forward(model):
    varlen_attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in VARLEN_ATTN_DISPATCH_MAPPING:
            if varlen_attn_forward is None:
                varlen_attn_forward = VARLEN_ATTN_DISPATCH_MAPPING[name]
                varlen_attn_forward = varlen_attn_forward.build()
            module.forward = types.MethodType(varlen_attn_forward, module)


# hack: replace embedding
def replace_embed(model):
    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if cls_name in EMBED_REPLACE_MAPPING:
                embed = EMBED_REPLACE_MAPPING[cls_name]
                embed = embed.build()
                child_new = embed(
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    padding_idx=child.padding_idx,
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


# hack: replace norm
def replace_norm(model):
    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if cls_name in NORM_REPLACE_MAPPING:
                norm = NORM_REPLACE_MAPPING[cls_name]
                norm = norm.build()
                child_new = norm(
                    norm_type=NORM2NEW_NORM_NAME_MAPPING[name],
                    normalized_shape=child.weight.shape,
                    eps=child.variance_epsilon,
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


# hack: replace linear
def replace_linear(model):
    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if cls_name in LINEAR_REPLACE_MAPPING:
                linear = LINEAR_REPLACE_MAPPING[cls_name]
                linear = linear.build()
                child_new = linear(
                    name=LINEAR2NEW_LINEAR_NAME_MAPPING.get(name, name),
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


# unified hack API: dispatch and replace modules
def dispatch_modules(model, use_packed_dataset):
    def check(model_name):
        assert "ForCausalLM" in model_name
        msg = "{} requires transformers version at least {}, but got {}"
        if model_name in LOWEST_TRANSFORMERS_VERSION:
            assert TRANSFORMERS_VERSION >= LOWEST_TRANSFORMERS_VERSION[model_name], msg.format(
                model_name, LOWEST_TRANSFORMERS_VERSION[model_name], TRANSFORMERS_VERSION
            )

    check(type(model).__name__)

    if use_packed_dataset:
        dispatch_varlen_attn_forward(model)
    else:
        dispatch_attn_forward(model)

    replace_embed(model)

    replace_norm(model)

    replace_linear(model)


__all__ = ["dispatch_modules"]
