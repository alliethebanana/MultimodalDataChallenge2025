"""

Model config


"""



from dataclasses import dataclass



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

    unknown_as_token: bool

    metadata_embedding_type: MetaDataEmbeddingConfig

    metadata_embedding_model_before_comb: str 

    combination_type: str 

    size_after_combination: int

    classifier_after_combination: str 


