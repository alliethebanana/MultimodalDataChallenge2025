"""

For getting specific configs 

"""

from src.config.model_config import ModelConfig, MetaDataEmbeddingConfig


def make_and_save_default_config():
    """
    Make the first default config
    """
    metadata_config = MetaDataEmbeddingConfig(
        habitat='default', location='default', substrate='default', event_date='default')
    
    




