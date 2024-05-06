# Define the classification model
import torch
from torch import nn
from vision_sinkformer import vit_small
from copy import deepcopy


class DINOClassificationModel(nn.Module):
    def __init__(self, hidden_size, num_classes, pretrained):
        """
        Load the pretrained DINOv2 Classification Model
        """
        # Initialize module
        super(DINOClassificationModel, self).__init__()

        # Load model with register
        model = vit_small(patch_size=14,
                          img_size=526,
                          init_values=1.0,
                          num_register_tokens=4,
                          block_chunks=0)
        self.embedding_size = 384
        self.number_of_heads = 6
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Load the pre-trained weights
        '''
        model.load_state_dict(
            torch.load(
                pretrained
            )
        )
        '''

        # Copy the model
        self.transformers = deepcopy(model)

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