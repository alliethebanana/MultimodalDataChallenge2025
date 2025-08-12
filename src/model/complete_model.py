"""

Model incorporating all steps

"""


import torch
import torch.nn as nn

from torchvision import models

from src.config.model_config import ModelConfig, MetaDataEmbeddingConfig
from src.util import convert_int_targets_to_one_hot


class CompleteModel(nn.Module):
    """ 
    Model made to fit the model class we consider in the article
    """
    def __init__(self, model_config: ModelConfig, num_targets: int = 183):
        super().__init__()
        
        self.model_config = model_config
        self.num_targets = num_targets
        self.possible_targets = torch.arange(self.num_targets)
        self.possible_targets_oh = convert_int_targets_to_one_hot(
            self.possible_targets, self.num_targets)
    

    def build_model(self):
        """
        Set the embedding and combination methods
        """
        match(self.model_config.image_embedding_type):
            case 'default':
                self.image_embedding = models.efficientnet_b0(pretrained=True)
            case _:
                raise ValueError(f'image embedding type not recognized: {self.model_config.image_embedding_type}')

        match(self.model_config.classifier_after_combination):
            case 'default':
                self.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.model_config.size_after_combination, self.num_targets)
                    )


    def predict_targets(self, image, metadata, device):
        """ Predicting targets from the image and metadata """


        




