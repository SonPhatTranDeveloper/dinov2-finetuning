# Define the classification result
import torch
from torch import nn
from vision_sinkformer import (vit_small_sinkhorn as vit_sink,
                               vit_small_weighted as vit_weighted,
                               vit_small_weighted_learnable as vit_weighted_learnable)
from dinov2.vision_transformer import vit_small
from copy import deepcopy


def create_vit_sinkformers(mode="weighted"):
    # Load pretrained result
    vit_transformers = vit_small(patch_size=14,
                                 img_size=526,
                                 init_values=1.0,
                                 num_register_tokens=4,
                                 block_chunks=0)
    vit_transformers.load_state_dict(
        torch.load("pretraied/dinov2_vits14_reg4_pretrain.pth")
    )

    # Create sinkformers result
    if mode == "sink":
        vit_sinkformers = vit_sink(patch_size=14,
                                   img_size=526,
                                   init_values=1.0,
                                   num_register_tokens=4,
                                   block_chunks=0)
    elif mode == "weighted":
        vit_sinkformers = vit_weighted(patch_size=14,
                                       img_size=526,
                                       init_values=1.0,
                                       num_register_tokens=4,
                                       block_chunks=0)
    elif mode == "weighted_learnable":
        vit_sinkformers = vit_weighted_learnable(patch_size=14,
                                                 img_size=526,
                                                 init_values=1.0,
                                                 num_register_tokens=4,
                                                 block_chunks=0)

    # Load the weight
    vit_sinkformers.load_state_dict(vit_transformers.state_dict(), strict=False)

    '''
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
    '''

    for block_sinkformer, block_transformer in zip(vit_sinkformers.blocks, vit_transformers.blocks):
        '''
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
        '''

        block_sinkformer.load_state_dict(block_transformer.state_dict(), strict=False)

        # Get the attention module
        attn_sinkformer = block_sinkformer.attn
        attn_transformer = block_transformer.attn

        # Copy the weight
        if mode == "sink":
            # Load Sinkhorn weight
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
        elif mode == "weighted" or mode == "weighted_learnable":
            # Load Sinkhorn and attention weight
            attn_sinkformer.softmax_attn.qkv.load_state_dict(
                attn_transformer.qkv.state_dict()
            )
            attn_sinkformer.softmax_attn.attn_drop.load_state_dict(
                attn_transformer.attn_drop.state_dict()
            )
            attn_sinkformer.softmax_attn.proj.load_state_dict(
                attn_transformer.proj.state_dict()
            )
            attn_sinkformer.softmax_attn.proj_drop.load_state_dict(
                attn_transformer.proj_drop.state_dict()
            )

            attn_sinkformer.sinkhorn_attn.qkv.load_state_dict(
                attn_transformer.qkv.state_dict()
            )
            attn_sinkformer.sinkhorn_attn.attn_drop.load_state_dict(
                attn_transformer.attn_drop.state_dict()
            )
            attn_sinkformer.sinkhorn_attn.proj.load_state_dict(
                attn_transformer.proj.state_dict()
            )
            attn_sinkformer.sinkhorn_attn.proj_drop.load_state_dict(
                attn_transformer.proj_drop.state_dict()
            )

    model = deepcopy(vit_sinkformers)
    return model


class DINOClassificationModel(nn.Module):
    def __init__(self, hidden_size, num_classes, mode="weighted"):
        """
        Load the pretrained DINOv2 Classification Model
        """
        # Initialize module
        super(DINOClassificationModel, self).__init__()

        # Load result with register
        self.embedding_size = 384
        self.number_of_heads = 6
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Copy the result
        self.transformers = create_vit_sinkformers(mode=mode)

        # Add the classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def adjust_sinkhorn_weight(self, new_weight):
        print("Adjust Sinkhorn Weight to: " + str(new_weight))
        for block_sinkformer in self.transformers.blocks:
            block_sinkformer.attn.sinkhorn_weight = new_weight

    def forward(self, inputs):
        """
        Forward the inputs
        inputs: tensor of size (batch_size, image_height, image_width, channels)
        """
        # Pass through the transformers and normalization
        outputs = self.transformers(inputs)
        outputs = self.transformers.norm(outputs)
        outputs = self.classifier(outputs)
        return outputs