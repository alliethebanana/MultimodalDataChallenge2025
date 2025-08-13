"""

For getting specific configs 

"""


from src import save_load_json

from src.config.model_config import ModelConfig, MetaDataEmbeddingConfig


def make_and_save_default_config():
    """
    Make the first default config
    """
    metadata_config = MetaDataEmbeddingConfig(
        habitat='default', location='default', substrate='default', event_date='default')
    
    model_config = ModelConfig(
        random_seed=0, 
        image_embedding_type='default',
        image_embedding_size=200,
        unknown_as_token=True,
        metadata_embedding_type=metadata_config,
        metadata_embedding_model_before_comb='none',
        combination_type='concat',
        classifier_after_combination='default',
        patience=5)
    
    config_path = 'configs/default_model_config.json'
    
    save_load_json.save_as_json(model_config, config_path)
    

def make_and_save_linear_before_comb_config():
    """
    Make config
    """
    metadata_config = MetaDataEmbeddingConfig(
        habitat='default', location='default', substrate='default', event_date='default')
    
    model_config = ModelConfig(
        random_seed=0, 
        image_embedding_type='default',
        image_embedding_size=200,
        unknown_as_token=True,
        metadata_embedding_type=metadata_config,
        metadata_embedding_model_before_comb='linear',
        combination_type='concat',
        classifier_after_combination='default',
        patience=5)
    
    config_path = 'configs/linear_b_comb_model_config.json'
    
    save_load_json.save_as_json(model_config, config_path)


def make_and_save_linear_before_comb_add_config():
    """
    Make config
    """
    metadata_config = MetaDataEmbeddingConfig(
        habitat='default', location='default', substrate='default', event_date='default')
    
    model_config = ModelConfig(
        random_seed=0, 
        image_embedding_type='default',
        image_embedding_size=200,
        unknown_as_token=True,
        metadata_embedding_type=metadata_config,
        metadata_embedding_model_before_comb='linear',
        combination_type='add',
        classifier_after_combination='default',
        patience=5)
    
    config_path = 'configs/linear_b_comb_add_model_config.json'
    
    save_load_json.save_as_json(model_config, config_path)


