"""

Model incorporating all steps

"""


import pandas as pd

import torch
import torch.nn as nn

from torchvision import models

from src.config.model_config import ModelConfig, MetaDataEmbeddingConfig
from src.data import metadata_util
from src.util import convert_int_targets_to_one_hot, get_month_from_date


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
        
        num_habitats, num_substrates = metadata_util.get_num_habitats_substrates()

        if self.model_config.unknown_as_token:
            num_habitats = num_habitats + 1
            num_substrates = num_substrates + 1 
        
        self.num_habitat_classes = num_habitats
        self.num_substrate_classes = num_substrates 

    

    def build_model(self):
        """
        Set the embedding and combination methods
        """
        # Image embedding type
        match(self.model_config.image_embedding_type):
            case 'default':
                self.image_embedding = models.efficientnet_b0(pretrained=True)
                self.image_embedding_size = self.image_embedding.classifier[1].out_features
            case _:
                raise ValueError(f'image embedding type not recognized: {self.model_config.image_embedding_type}')

        # Metadata embedding types
        md_emb_type = self.model_config.metadata_embedding_type
        match(md_emb_type.event_date):
            case 'default':
                self.date_emb = get_month_from_date
            case _:
                raise ValueError(f'event_date embedding type not recognized: {md_emb_type.event_date}')
        
        match(md_emb_type.habitat):
            case 'default':
                self.habitat_emb = lambda x: convert_int_targets_to_one_hot(
                    metadata_util.translate_habitats_to_class_labels(x))
            case _:
                raise ValueError(f'habitat embedding type not recognized: {md_emb_type.habitat}')
        
        match(md_emb_type.substrate):
            case 'default':
                self.substrate_emb = lambda x: convert_int_targets_to_one_hot(
                    metadata_util.translate_substrate_to_class_labels(x))
            case _:
                raise ValueError(f'substrate embedding type not recognized: {md_emb_type.substrate}')
        
        match(md_emb_type.location):
            case 'default':
                self.location_emb = lambda x, y: torch.concatenate([x, y], dim = 0)
            case _:
                raise ValueError(f'location embedding type not recognized: {md_emb_type.location}')
        
        
        # Combination type
        match(self.model_config.combination_type):
            case 'concat':
                self.comb_type = lambda x, y: torch.concatenate([x, y])
            case 'dot':
                self.comb_type = lambda x, y: torch.dot(x, y)
            case _:
                raise ValueError(f'combination type not recognized: {self.model_config.combination_type}')


        # Classifier to use after the combination
        match(self.model_config.classifier_after_combination):
            case 'default':
                self.classifier = nn.Sequential(
                    nn.Dropout(0.2, inplace=True),
                    nn.Linear(self.model_config.size_after_combination, 1000, bias=True),
                    nn.Dropout(0.2),
                    nn.Linear(1000, self.num_targets, bias = True)
                    )
            case _:
                raise ValueError(f'classifier type not recognized: {self.model_config.classifier_after_combination}')


    def predict_targets(self, image, metadata_df: pd.DataFrame, device):
        """ Predicting targets from the image and metadata """
        embedded_image = self.image_embedding(image)
        embedded_date = self.date_emb(metadata_df['eventDate'].values)
        embedded_habitat = self.habitat_emb(metadata_df['Habitat'])
        embedded_substrate = self.substrate_emb(metadata_df['Substrate'])
        embedding_location = self.location_emb(
            metadata_df['Latitude'].values, metadata_df['Longitude'].values)
        
        embedded_metadata = torch.concat(
            [embedded_date, embedded_habitat, embedded_substrate, embedding_location])
        
        combined_embedding = self.comb_type(embedded_image, embedded_metadata)

        output = self.classifier(combined_embedding)

        return output
