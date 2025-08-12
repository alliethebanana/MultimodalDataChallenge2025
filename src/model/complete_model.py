"""

Model incorporating all steps

"""


import torch
import torch.nn as nn

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
        

        




