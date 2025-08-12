"""

For getting specific configs 

"""

from src.config.model_config import ModelConfig, MetaDataEmbeddingConfig


def get_default_config():
    """
    Get the first default config
    """
    metadata_config = MetaDataEmbeddingConfig(
        habitat='default', location='default', substrate='default', event_date='default')
    
    




