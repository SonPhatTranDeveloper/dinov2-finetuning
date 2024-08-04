from dinov2.vision_transformer import DinoVisionTransformer
from dinov2.layers import NestedTensorBlock as Block
from functools import partial
from dinov2.layers.sinkhorn import (ScaledProductAttentionSinkhorn,
                                    WeightedCombinationAttention,
                                    WeightedLearnableAttention)


# Create small vision transformer model
def vit_small_sinkhorn(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=ScaledProductAttentionSinkhorn),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


# Create small vision transformer model
def vit_small_weighted(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=WeightedCombinationAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_small_weighted_learnable(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=WeightedLearnableAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


