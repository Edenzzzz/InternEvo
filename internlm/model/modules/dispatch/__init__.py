# adapted from https://github.com/InternLM/xtuner/blob/main/xtuner/model/modules/dispatch/__init__.py

from internlm.core.context import global_context as gpc
from internlm.model.modules.dispatch.utils import LazyObject


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


# unified hack API: dispatch model
def dispatch_model(model):

    replace_embed(model)

    replace_norm(model)

    replace_linear(model)


# unified hack API: dispatch config
def dispatch_config(config):
    gpc.config.model.vocab_size = gpc.config.VOCAB_SIZE = config.vocab_size
    gpc.config.model.hidden_size = gpc.config.HIDDEN_SIZE = config.hidden_size
    gpc.config.model.num_layers = gpc.config.NUM_LAYER = config.num_hidden_layers
    gpc.config.model.num_attention_heads = gpc.config.NUM_ATTENTION_HEAD = config.num_attention_heads
    gpc.config.model.mlp_ratio = gpc.config.MLP_RATIO = config.intermediate_size / config.hidden_size

    # For models that use GQA
    if hasattr(config, "num_key_value_heads"):
        gpc.config.model.num_kv_attention_heads = gpc.config.NUM_KV_ATTENTION_HEAD = config.num_key_value_heads


__all__ = ["dispatch_model", "dispatch_config"]
