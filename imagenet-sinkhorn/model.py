# Define the classification model
import torch
from torch import nn
from vision_sinkformer import vit_small as vit_sink
from dinov2.vision_transformer import vit_small
from copy import deepcopy


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

    # Create sinkformers model
    vit_sinkformers = vit_sink(patch_size=14,
                               img_size=526,
                               init_values=1.0,
                               num_register_tokens=4,
                               block_chunks=0)

    # Copy the weights
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


class DINOClassificationModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        """
        Load the pretrained DINOv2 Classification Model
        """
        # Initialize module
        super(DINOClassificationModel, self).__init__()

        # Load model with register
        self.embedding_size = 384
        self.number_of_heads = 6
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Copy the model
        self.transformers = create_vit_sinkformers()

        # Add the classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

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