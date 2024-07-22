import torch
from vision_sinkformer import vit_small_sinkhorn as vit_sink
from dinov2.vision_transformer import vit_small
from copy import deepcopy


# Test result loading function
def create_vit_sinkformers():
    # Load pretrained model
    vit_transformers = vit_small(patch_size=14,
                                 img_size=526,
                                 init_values=1.0,
                                 num_register_tokens=4,
                                 block_chunks=0)
    vit_transformers.load_state_dict(
        torch.load("pretraied/dinov2_vits14_reg4_pretrain.pth")
    )

    # Create sinkformers result
    vit_sinkformers = vit_sink(patch_size=14,
                               img_size=526,
                               init_values=1.0,
                               num_register_tokens=4,
                               block_chunks=0)

    # Copy and freeze the weights
    vit_sinkformers.patch_embed.load_state_dict(
        vit_transformers.patch_embed.state_dict()
    )

    vit_sinkformers.norm.load_state_dict(
        vit_transformers.norm.state_dict()
    )

    vit_sinkformers.head.load_state_dict(
        vit_transformers.head.state_dict()
    )

    for block_sinkformer, block_transformer in zip(vit_sinkformers.blocks, vit_transformers.blocks):
        # Load the other weights
        block_sinkformer.norm1.load_state_dict(
            block_transformer.norm1.state_dict()
        )

        block_sinkformer.ls1.load_state_dict(
            block_transformer.ls1.state_dict()
        )

        block_sinkformer.drop_path1.load_state_dict(
            block_transformer.drop_path1.state_dict()
        )

        block_sinkformer.norm2.load_state_dict(
            block_transformer.norm2.state_dict()
        )

        block_sinkformer.mlp.load_state_dict(
            block_transformer.mlp.state_dict()
        )

        block_sinkformer.ls2.load_state_dict(
            block_transformer.ls2.state_dict()
        )

        block_sinkformer.drop_path2.load_state_dict(
            block_transformer.drop_path2.state_dict()
        )

        # Get the attention module
        attn_sinkformer = block_sinkformer.attn
        attn_transformer = block_transformer.attn

        # Copy the weight
        attn_sinkformer.qkv.load_state_dict(
            attn_transformer.qkv.state_dict()
        )
        attn_sinkformer.attn_drop.load_state_dict(
            attn_transformer.attn_drop.state_dict()
        )
        attn_sinkformer.proj.load_state_dict(
            attn_transformer.proj.state_dict()
        )
        attn_sinkformer.proj_drop.load_state_dict(
            attn_transformer.proj_drop.state_dict()
        )

    model = deepcopy(vit_sinkformers)
    return model


if __name__ == "__main__":
    create_vit_sinkformers()