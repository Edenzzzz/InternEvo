# adapted from https://github.com/InternLM/xtuner/blob/main/xtuner/model/modules/dispatch/__init__.py

# Copyright (c) OpenMMLab. All rights reserved.
import types
import transformers

from packaging.version import Version
from transformers.utils.import_utils import is_flash_attn_2_available

from internlm.model.modules.dispatch.utils import LazyObject

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


__all__ = ["dispatch_modules"]
