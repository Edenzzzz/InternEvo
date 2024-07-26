# adapted from https://github.com/InternLM/xtuner/blob/main/xtuner/model/modules/dispatch/internlm.py

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from typing import Optional, Tuple

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.modules.dispatch.triton_kernels import apply_rotary_emb

SUPPORT_FLASH2 = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    SUPPORT_FLASH2 = True
except ImportError:
    pass


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def internlm_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,  # pylint: disable=W0613
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

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

    attn_output = self.o_proj(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


def internlm_varlen_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,  # pylint: disable=W0613
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,  # pylint: disable=W0613
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    cu_seqlens = gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
    max_seqlen = gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]

    use_varlen_atten = cu_seqlens is not None

    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1, f"If utilizing local attention, the batch size should be" f" set to 1, but got {bsz}"

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)

    kv_seq_len = key_states.shape[-3]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if use_varlen_atten:
        cos, sin = self.rotary_emb(value_states, max_seqlen)
        query_states = apply_rotary_emb(query_states, cos[position_ids].squeeze(0), sin[position_ids].squeeze(0))
        key_states = apply_rotary_emb(key_states, cos[position_ids].squeeze(0), sin[position_ids].squeeze(0))
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    assert SUPPORT_FLASH2
    if use_varlen_atten:
        q_unpad, k_unpad, v_unpad = query_states.flatten(0, 1), key_states.flatten(0, 1), value_states.flatten(0, 1)
        attn_output = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
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

    attn_output = self.o_proj(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value
