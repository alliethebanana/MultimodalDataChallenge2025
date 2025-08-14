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
    

def make_and_save_no_meta_config():
    """
    Make config
    """
    metadata_config = None

    model_config = ModelConfig(
        random_seed=0, 
        image_embedding_type='default',
        image_embedding_size=500,
        unknown_as_token=True,
        metadata_embedding_type=metadata_config,
        metadata_embedding_model_before_comb='none',
        combination_type='concat',
        classifier_after_combination='default',
        patience=5)
    
    config_path = 'configs/no_meta_model_config.json'
    
    save_load_json.save_as_json(model_config, config_path)


def make_and_save_no_meta_dino_config():
    """
    Make config
    """
    metadata_config = None

    model_config = ModelConfig(
        random_seed=0, 
        image_embedding_type='dino',
        image_embedding_size=384,
        unknown_as_token=True,
        metadata_embedding_type=metadata_config,
        metadata_embedding_model_before_comb='none',
        combination_type='concat',
        classifier_after_combination='default',
        patience=5)
    
    config_path = 'configs/no_meta_dino_model_config.json'
    
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


def make_and_save_linear_before_comb_mlp_classifier_config():
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
        classifier_after_combination='mlp',
        patience=5)
    
    config_path = 'configs/linear_b_comb_mlp_class_model_config.json'
    
    save_load_json.save_as_json(model_config, config_path)


def make_and_save_cyclical_fourier_config():
    """
    Make config
    """
    metadata_config = MetaDataEmbeddingConfig(
        habitat='default', location='fourier', substrate='default', event_date='cyclic_month')
    
    model_config = ModelConfig(
        random_seed=0, 
        image_embedding_type='dino',
        image_embedding_size=384,
        unknown_as_token=True,
        metadata_embedding_type=metadata_config,
        metadata_embedding_model_before_comb='none',
        combination_type='concat',
        classifier_after_combination='mlp',
        patience=5)
    
    config_path = 'configs/cyclical_fourier_config.json'
    
    save_load_json.save_as_json(model_config, config_path)

    
def make_and_save_linear_cyclical_fourier_config():
    """
    Make config
    
    """
        
    metadata_config = MetaDataEmbeddingConfig(
        habitat='default', location='fourier', substrate='default', event_date='cyclic_month')
    
    model_config = ModelConfig(
        random_seed=0, 
        image_embedding_type='dino',
        image_embedding_size=384,
        unknown_as_token=True,
        metadata_embedding_type=metadata_config,
        metadata_embedding_model_before_comb='linear',
        combination_type='concat',
        classifier_after_combination='mlp',
        patience=5)
    
    config_path = 'configs/linear_linear_cyclical_fourier_config.json'
    
    save_load_json.save_as_json(model_config, config_path)


def make_and_save_mlp_norm_cyclical_fourier_config():
    """
    Make config
    
    """
        
    metadata_config = MetaDataEmbeddingConfig(
        habitat='default', location='fourier', substrate='default', event_date='cyclic_month')
    
    model_config = ModelConfig(
        random_seed=0, 
        image_embedding_type='dino',
        image_embedding_size=384,
        unknown_as_token=True,
        metadata_embedding_type=metadata_config,
        metadata_embedding_model_before_comb='mlp_norm',
        combination_type='concat',
        classifier_after_combination='mlp',
        patience=5)
    
    config_path = 'configs/mlp_norm_cyclical_fourier_config.json'
    
    save_load_json.save_as_json(model_config, config_path)

    def make_and_save_mlp_norm_cyclical_fourier_clip_config():
        """
        Make config
        
        """
            
        metadata_config = MetaDataEmbeddingConfig(
            habitat='clip', location='fourier', substrate='clip', event_date='cyclic_month')
        
        model_config = ModelConfig(
            random_seed=0, 
            image_embedding_type='dino',
            image_embedding_size=384,
            unknown_as_token=True,
            metadata_embedding_type=metadata_config,
            metadata_embedding_model_before_comb='mlp_norm',
            combination_type='concat',
            classifier_after_combination='mlp',
            patience=5)
        
        config_path = 'configs/mlp_norm_cyclical_fourier_clip_config.json'
        
        save_load_json.save_as_json(model_config, config_path)