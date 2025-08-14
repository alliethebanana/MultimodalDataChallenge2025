"""

Model incorporating all steps

"""

from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
import open_clip

from torchvision import models

from src.config.model_config import ModelConfig
from src.data import metadata_util
from src.model.mlp import SmallMLP
from src.model.MetadataMLP import MetadataMLP
from src.util import convert_int_targets_to_one_hot


from src.model.helpers.CyclicMonth import CyclicMonth
from src.model.helpers.FourierLatLon import FourierLatLon

def clip_encoding_habitat(data,num_classes,model,tokenizer):
    num_to_string = {0:'null',
                    1:'Mixed woodland (with coniferous and deciduous trees)',
                    2:'Unmanaged deciduous woodland', 
                    3:'Forest bog',
                    4:'coniferous woodland/plantation', 
                    5:'Deciduous woodland',
                    6:'natural grassland', 
                    7:'lawn', 
                    8:'Unmanaged coniferous woodland',
                    9:'garden',
                    10:'wooded meadow, grazing forest',
                    11:'dune',
                    12:'Willow scrubland',
                    13:'heath',
                    14:'Acidic oak woodland',
                    15:'roadside',
                    16:'Thorny scrubland',
                    17:'park/churchyard',
                    18:'Bog woodland',
                    19:'hedgerow',
                    20:'gravel or clay pit',
                    21:'salt meadow',
                    22:'bog',
                    23:'meadow',
                    24:'improved grassland',
                    25:'other habitat',
                    26:'roof',
                    27:'fallow field',
                    28:'ditch',
                    29:'fertilized field in rotation'}
    data = [num_to_string[int(x)] for x in data]
    text = tokenizer(data)
    text = text.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features

def clip_encoding_substrate(data,num_classes,model,tokenizer):
    num_to_string = {0:'null',
    1:'soil',
    2:'leaf or needle litter',
    3:'wood chips or mulch',
    4:'dead wood (including bark)',
    5:'bark',
    6:'wood',
    7:'bark of living trees',
    8:'mosses',
    9:'wood and roots of living trees',
    10:'stems of herbs, grass etc',
    11:'peat mosses',
    12:'dead stems of herbs, grass etc',
    13:'fungi',
    14:'other substrate',
    15:'living stems of herbs, grass etc',
    16:'living leaves',
    17:'fire spot',
    18:'faeces',
    19:'cones',
    20:'fruits',
    21:'catkins'}
    data = [num_to_string[int(x)] for x in data]
    text = tokenizer(data)
    text = text.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features

    

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

        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.build_model()

    
    def build_model(self):
        """
        Set the embedding and combination methods
        """
        # Image embedding type
        match(self.model_config.image_embedding_type):
            case 'default':
                self.image_embedding = models.efficientnet_b0(
                    pretrained=True)
                self.image_embedding.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(
                        self.image_embedding.classifier[1].in_features, 
                        self.model_config.image_embedding_size
                ))
                self.image_embedding_size = self.model_config.image_embedding_size
            case 'dino':
                self.image_embedding = lambda x: x
                self.image_embedding_size = 384
            case _:
                raise ValueError(
                    f'image embedding type not recognized: {self.model_config.image_embedding_type}')

        metadata_emb_size = 0
        md_emb_type = self.model_config.metadata_embedding_type
        if md_emb_type is None:
            self.size_after_combination = self.image_embedding_size
            self.metadata_emb_size = metadata_emb_size
        
        else:
            # Metadata embedding types        
            match(md_emb_type.event_date):
                case 'default':
                    self.date_emb = lambda x: x
                    metadata_emb_size += 1
                case 'cyclic_month':
                    self.date_emb = CyclicMonth()      # -> (N, 2)
                    date_emb_size = 2
                    metadata_emb_size += date_emb_size
                case _:
                    raise ValueError(
                        f'event_date embedding type not recognized: {md_emb_type.event_date}')
            
            match(md_emb_type.habitat):
                case 'default':
                    self.habitat_emb = convert_int_targets_to_one_hot
                    metadata_emb_size += self.num_habitat_classes
                case 'clip':
                    self.habitat_emb = partial(clip_encoding_habitat,model=self.clip,tokenizer=self.tokenizer)
                    metadata_emb_size += 512
                    
                case _:
                    raise ValueError(
                        f'habitat embedding type not recognized: {md_emb_type.habitat}')
            
            match(md_emb_type.substrate):
                case 'default':
                    self.substrate_emb = convert_int_targets_to_one_hot
                    metadata_emb_size += self.num_substrate_classes
                case 'clip':
                    self.substrate_emb = partial(clip_encoding_substrate,model=self.clip,tokenizer=self.tokenizer)
                    metadata_emb_size += 512
                case _:
                    raise ValueError(
                        f'substrate embedding type not recognized: {md_emb_type.substrate}')
            
            match(md_emb_type.location):
                case 'fourier':
                    self.location_emb = FourierLatLon()
                    metadata_emb_size += 4 * self.location_emb.n_freqs
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
                case 'mlp_norm':  #LayerNorm→Dropout→MLP (residual)
                    self.before_comb_model = MetadataMLP(
                        in_dim=self.metadata_emb_size,
                        target_dim=self.image_embedding_size,
                        hidden=512,         
                        p_drop=0.3           
                    )
                    self.metadata_emb_size = self.image_embedding_size
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
                case 'add':
                    self.linear_projection = nn.Linear(
                            self.metadata_emb_size, 
                            self.image_embedding_size
                    )
                    self.comb_type = self.add_reps
                    self.size_after_combination = self.image_embedding_size
                case _:
                    raise ValueError(
                        f'combination type not recognized: {self.model_config.combination_type}')


        # Classifier to use after the combination
        match(self.model_config.classifier_after_combination):
            case 'default':
                self.classifier = nn.Sequential(
                    nn.Linear(self.size_after_combination, 500, bias=True),
                    nn.Dropout(0.2),
                    nn.Linear(500, self.num_targets, bias = True)
                    )
            case 'mlp':
                mlp_model = SmallMLP(
                    self.size_after_combination, 
                    final_dim=400,
                    num_features=512,
                    nonlinearity= 'leaky_relu')
                self.classifier = nn.Sequential(
                    mlp_model,
                    nn.Dropout(0.1),
                    nn.Linear(400, self.num_targets, bias = True)
                    )
            case _:
                raise ValueError(
                    f'classifier type not recognized: {self.model_config.classifier_after_combination}')


    def add_reps(self, img, md):
        """ Project metadata to same size as image embedding and add """
        projected_md = self.linear_projection(md)
        return img + projected_md


    def forward(self, image, metadata: Tuple, device):
        """ Predicting targets from the image and metadata """
        embedded_image = self.image_embedding(image)
        
        if self.metadata_emb_size == 0:
            combined_embedding = embedded_image
        else:
            dates, habitats, substrates, locations = metadata
            dates, habitats = dates.to(device), habitats.to(device)
            substrates, locations = substrates.to(device), locations.to(device)

            
            embedded_date = self.date_emb(dates)
            embedded_habitat = self.habitat_emb(habitats, self.num_habitat_classes).to(device)
            embedded_substrate = self.substrate_emb(substrates, self.num_substrate_classes).to(device)
            embedding_location = self.location_emb(locations)
            
            embedded_metadata = torch.concat(
                [embedded_date, embedded_habitat, embedded_substrate, embedding_location], dim = 1)
            embedded_metadata = embedded_metadata.float()
            embedded_metadata = self.before_comb_model(embedded_metadata)
            
            combined_embedding = self.comb_type(embedded_image, embedded_metadata)
            combined_embedding = combined_embedding.float()

        output = self.classifier(combined_embedding)

        return output
    

    def predict(self, image, metadata: Tuple, device):
        """ Predicting targets from the image and metadata """
        output = self.forward(image, metadata, device)
        predictions = torch.nn.Softmax(dim=-1)(output)

        return predictions
