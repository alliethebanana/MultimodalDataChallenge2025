"""

Model incorporating all steps

"""

from typing import Tuple

import torch
import torch.nn as nn

from torchvision import models

from src.config.model_config import ModelConfig
from src.data import metadata_util
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
        
        num_habitats, num_substrates = metadata_util.get_num_habitats_substrates()

        if self.model_config.unknown_as_token:
            num_habitats = num_habitats + 1
            num_substrates = num_substrates + 1 
        
        self.num_habitat_classes = num_habitats
        self.num_substrate_classes = num_substrates 

        self.build_model()

    
    def build_model(self):
        """
        Set the embedding and combination methods
        """
        # Image embedding type
        match(self.model_config.image_embedding_type):
            case 'default':
                self.image_embedding = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                self.image_embedding.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(
                        self.image_embedding.classifier[1].in_features, 
                        self.model_config.image_embedding_size
                ))
                self.image_embedding_size = self.model_config.image_embedding_size
            case _:
                raise ValueError(
                    f'image embedding type not recognized: {self.model_config.image_embedding_type}')

        # Metadata embedding types
        metadata_emb_size = 0
        md_emb_type = self.model_config.metadata_embedding_type
        match(md_emb_type.event_date):
            case 'default':
                self.date_emb = lambda x: x
                metadata_emb_size += 1
            case _:
                raise ValueError(
                    f'event_date embedding type not recognized: {md_emb_type.event_date}')
        
        match(md_emb_type.habitat):
            case 'default':
                self.habitat_emb = convert_int_targets_to_one_hot
                metadata_emb_size += self.num_habitat_classes
            case _:
                raise ValueError(
                    f'habitat embedding type not recognized: {md_emb_type.habitat}')
        
        match(md_emb_type.substrate):
            case 'default':
                self.substrate_emb = convert_int_targets_to_one_hot
                metadata_emb_size += self.num_substrate_classes
            case _:
                raise ValueError(
                    f'substrate embedding type not recognized: {md_emb_type.substrate}')
        
        match(md_emb_type.location):
            case 'default':
                self.location_emb = lambda x: x
                metadata_emb_size += 2
            case _:
                raise ValueError(
                    f'location embedding type not recognized: {md_emb_type.location}')
        
        self.metadata_emb_size = metadata_emb_size

        # metadata model before combination
        match(self.model_config.metadata_embedding_model_before_comb):
            case 'none':
                self.before_comb_model = lambda x: x 
            case 'linear':
                self.before_comb_model = nn.Linear(
                        self.metadata_emb_size, 
                        self.image_embedding_size
                )
                self.metadata_emb_size = self.image_embedding_size
            case _:
                raise ValueError(
                    f'metadata_embedding_model_before_comb type not recognized: {self.model_config.metadata_embedding_model_before_comb}')
        
        # Combination type
        match(self.model_config.combination_type):
            case 'concat':
                self.comb_type = lambda x, y: torch.concatenate([x, y], dim = 1)
                self.size_after_combination = self.metadata_emb_size + self.image_embedding_size
            case 'dot':
                self.linear_projection = nn.Linear(
                        self.metadata_emb_size, 
                        self.image_embedding_size
                )
                self.comb_type = self.dot_reps
                self.size_after_combination = self.image_embedding_size
            case _:
                raise ValueError(
                    f'combination type not recognized: {self.model_config.combination_type}')


        # Classifier to use after the combination
        match(self.model_config.classifier_after_combination):
            case 'default':
                self.classifier = nn.Sequential(
                    nn.Linear(self.size_after_combination, 200, bias=True),
                    nn.Dropout(0.2),
                    nn.Linear(200, self.num_targets, bias = True)
                    )
            case _:
                raise ValueError(
                    f'classifier type not recognized: {self.model_config.classifier_after_combination}')


    def dot_reps(self, img, md):
        """ Project metadata to same size as image embedding and dot """
        projected_md = self.linear_projection(md)
        return torch.dot(img, projected_md)


    def forward(self, image, metadata: Tuple, device):
        """ Predicting targets from the image and metadata """
        dates, habitats, substrates, locations = metadata
        dates, habitats = dates.to(device), habitats.to(device)
        substrates, locations = substrates.to(device), locations.to(device)

        embedded_image = self.image_embedding(image)
        embedded_date = self.date_emb(dates)
        embedded_habitat = self.habitat_emb(habitats, self.num_habitat_classes)
        embedded_substrate = self.substrate_emb(substrates, self.num_substrate_classes)
        embedding_location = self.location_emb(locations)
        
        embedded_metadata = torch.concat(
            [torch.unsqueeze(embedded_date, 1), 
             embedded_habitat, embedded_substrate, embedding_location], dim = 1)
        embedded_metadata = self.before_comb_model(embedded_metadata)
        
        combined_embedding = self.comb_type(embedded_image, embedded_metadata)

        output = self.classifier(combined_embedding)

        return output
    

    def predict(self, image, metadata: Tuple, device):
        """ Predicting targets from the image and metadata """
        output = self.forward(image, metadata, device)
        predictions = torch.nn.Softmax(dim=-1)(output)

        return predictions
