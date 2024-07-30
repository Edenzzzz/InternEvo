# adapted from https://github.com/InternLM/xtuner/blob/main/xtuner/model/modules/dispatch/internlm2.py

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.modules.dispatch.common import apply_rotary_pos_emb
from transformers.cache_utils import Cache, StaticCache

SUPPORT_FLASH2 = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    SUPPORT_FLASH2 = True
except ImportError:
    pass


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv_bshd(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """The hidden states go from (batch, seqlen, num_key_value_heads, head_dim)
    to (batch, seqlen, num_attention_heads, head_dim)"""
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def internlm2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,  # pylint: disable=W0613
    use_cache: bool = False,  # pylint: disable=W0613
    cache_position: Optional[torch.LongTensor] = None,
):
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with "
            "`attn_implementation==flash_attention_2` make sure to use `sdpa` "
            "in the mean time, and open an issue at "
            "https://github.com/huggingface/transformers"
        )

    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models;
        # cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32. (InternLM2RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.wqkv.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    if SUPPORT_FLASH2:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        attn_output = attn_output.contiguous()
    else:
        # use flash attention implemented by pytorch
        attn_output = F.scaled_dot_product_attention(  # pylint: disable=E1102
            query_states, key_states, value_states, attn_mask=attention_mask
        )
        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


def internlm2_varlen_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,  # pylint: disable=W0613
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,  # pylint: disable=W0613
    use_cache: bool = False,  # pylint: disable=W0613
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with "
            "`attn_implementation==flash_attention_2` make sure to use `sdpa` "
            "in the mean time, and open an issue at "
            "https://github.com/huggingface/transformers"
        )

    cu_seqlens = gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
    max_seqlen = gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]

    use_varlen_atten = cu_seqlens is not None

    bsz, q_len, _ = hidden_states.size()

    assert bsz == 1, f"If utilizing local attention, the batch size should be" f" set to 1, but got {bsz}"

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    try:
        cos, sin = self.rotary_emb(value_states, position_ids.unsqueeze(0))
    except RuntimeError:
        raise RuntimeError(
            "You are using the old version of InternLM2 model. The "
            "`modeling_internlm2.py` is outdated. Please update the InternLM2 "
            "model."
        )
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models;
        # cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32. (InternLM2RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.wqkv.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # repeat kv for sequence parallel
    key_states = repeat_kv_bshd(key_states, self.num_key_value_groups)
    value_states = repeat_kv_bshd(value_states, self.num_key_value_groups)

    assert SUPPORT_FLASH2

    if use_varlen_atten:
        attn_output = flash_attn_varlen_func(
            query_states.flatten(0, 1),
            key_states.flatten(0, 1),
            value_states.flatten(0, 1),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            0,
            return_attn_probs=False,
            causal=True,
        )
    else:
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value
