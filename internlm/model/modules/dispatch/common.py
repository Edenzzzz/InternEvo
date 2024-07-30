import torch


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    if position_ids:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
