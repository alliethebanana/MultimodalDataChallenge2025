"""

Model config


"""



from dataclasses import dataclass

from src import save_load_json


@dataclass
class MetaDataEmbeddingConfig:
    """ Config class for a metadata embeddings """
    habitat: str # default
    location: str # default (latitude and longitude)
    substrate: str # default
    event_date: str # default


@dataclass
class ModelConfig:
    """ Config class for a model """
    random_seed: int
    
    image_embedding_type: str # default
    image_embedding_size: int

    unknown_as_token: bool

    metadata_embedding_type: MetaDataEmbeddingConfig

    metadata_embedding_model_before_comb: str 

    combination_type: str 

    classifier_after_combination: str 

    patience: int


def load_model_config(path: str):
    """ Load the model config json """
    json_dict = save_load_json.load_json(path) 

    if json_dict['metadata_embedding_type'] is not None:
        md_config = MetaDataEmbeddingConfig(**json_dict['metadata_embedding_type'])

        json_dict['metadata_embedding_type'] = md_config

    model_config = ModelConfig(**json_dict)

    return model_config
