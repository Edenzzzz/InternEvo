# adapted from https://github.com/InternLM/xtuner/blob/main/xtuner/model/modules/dispatch/__init__.py

# Copyright (c) OpenMMLab. All rights reserved.
import types

from packaging.version import Version

import transformers
from internlm.model.modules.dispatch.utils import LazyObject
from internlm.utils.parallel import is_using_isp
from transformers.utils.import_utils import is_flash_attn_2_available

try:
    import triton  # pre-check # noqa: F401
    import triton.language as tl  # pre-check # noqa: F401
except ImportError:
    raise ImportError("triton is not installed.")

SUPPORT_FLASH2 = is_flash_attn_2_available()

TRANSFORMERS_VERSION = Version(transformers.__version__)

LOWEST_TRANSFORMERS_VERSION = dict(
    InternLMForCausalLM=Version("4.36"),
)

ATTN_DISPATCH_MAPPING = dict(
    InternLMAttention=LazyObject("internlm.model.modules.dispatch.internlm", "internlm_attn_forward"),
)

VARLEN_ATTN_DISPATCH_MAPPING = dict(
    InternLMAttention=LazyObject("internlm.model.modules.dispatch.internlm", "internlm_varlen_attn_forward"),
)

RMS_DISPATCH_MAPPING = dict(
    InternLMRMSNorm=LazyObject("internlm.model.modules.dispatch.triton_kernels", "rms_norm_forward"),
)

ROTE_DISPATCH_MAPPING = dict()

EMBED_DISPATCH_MAPPING = dict(
    Embedding=LazyObject("internlm.model.modules.embedding", "Embedding1D"),
)

LINEAR_DISPATCH_MAPPING = dict(
    Linear=LazyObject("internlm.model.modules.linear", "new_linear"),
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


def dispatch_attn_forward(model):

    if not SUPPORT_FLASH2:
        return

    attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in ATTN_DISPATCH_MAPPING:
            if attn_forward is None:
                attn_forward = ATTN_DISPATCH_MAPPING[name]
                attn_forward = attn_forward.build()
            module.forward = types.MethodType(attn_forward, module)


def dispatch_varlen_attn_forward(model):

    if not SUPPORT_FLASH2:
        return

    varlen_attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in VARLEN_ATTN_DISPATCH_MAPPING:
            if varlen_attn_forward is None:
                varlen_attn_forward = VARLEN_ATTN_DISPATCH_MAPPING[name]
                varlen_attn_forward = varlen_attn_forward.build()
            module.forward = types.MethodType(varlen_attn_forward, module)


def dispatch_rmsnorm_forward(model):

    rms_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in RMS_DISPATCH_MAPPING:
            if rms_forward is None:
                rms_forward = RMS_DISPATCH_MAPPING[name]
                rms_forward = rms_forward.build()
            module.forward = types.MethodType(rms_forward, module)


def replace_rote(model):
    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if cls_name in ROTE_DISPATCH_MAPPING:
                assert hasattr(model.config, "rope_theta"), "`rope_theta` should be in the model config."
                rope_theta = model.config.rope_theta

                rote = ROTE_DISPATCH_MAPPING[cls_name]
                rote = rote.build()
                dim_model = child.inv_freq.shape[0] * 2
                child_new = rote(dim_model, child.max_seq_len_cached, rope_theta).to(
                    device=child.inv_freq.device, dtype=child.inv_freq.dtype
                )
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


def replace_embed(model):
    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if cls_name in EMBED_DISPATCH_MAPPING:
                embed = EMBED_DISPATCH_MAPPING[cls_name]
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


def replace_linear(model):
    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if cls_name in LINEAR_DISPATCH_MAPPING:
                linear = LINEAR_DISPATCH_MAPPING[cls_name]
                linear = linear.build()
                child_new = linear(
                    name=LINEAR2NEW_LINEAR_NAME_MAPPING[name],
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


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

    dispatch_rmsnorm_forward(model)

    replace_rote(model)

    if is_using_isp():
        replace_embed(model)
        replace_linear(model)


__all__ = ["dispatch_modules"]
