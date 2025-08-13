"""
Run Script for training fungi classification models


Usage:
    run.py train --checkpoint-folder=<file> --image-folder=<file> --metadata-folder=<file> --model-config=<file> [options]
    run.py evaluate --checkpoint-folder=<file> --image-folder=<file> --metadata-folder=<file> --model-config=<file> [options]
    
Options:
    -h --help                               show this screen.
    --checkpoint-folder=<file>              folder to save trained model(s) in
    --image-folder=<file>                   folder to load images from  
    --metadata-folder=<file>                folder to load meta data from 
    --model-config=<file>                   path to model config
    --session=<string>                      session name [default: EfficientNet]
    --log=<string>                          log level to use [default: info]  
    --train-seed=<int>                      if set, will overwrite the seed in the training config.[default: -1]
    --cuda                                  use GPU 
    
"""


import os

from datetime import datetime 

import logging

from typing import Dict

from docopt import docopt

from src.config.model_config import load_model_config

from src.training.train_fungi_network import train_fungi_network, evaluate_network_on_test_set



def train(args:Dict) -> None:
    """
    Train model  
    """   
    # Path to fungi images   
    # image_path = '/scratch/bmgi/FungiImages'
    image_path = args['--image-folder'] if args['--image-folder'] else ''

    # Path to metadata file
    # data_file = './starting_metadata'
    metadata_folder = args['--metadata-folder'] if args['--metadata-folder'] else ''

    # Session name: Change session name for every experiment! 
    # Session name will be saved as the first line of the prediction file
    session = args['--session'] if args['--session'] else "EfficientNet"

    # Folder for results of this experiment based on session name:
    # checkpoint_dir = "./results"
    checkpoint_dir = args['--checkpoint-folder'] if args['--checkpoint-folder'] else ''

    model_config_path = args['--model-config'] if args['--model-config'] else ''

    if image_path == '' or metadata_folder == '' or checkpoint_dir == '' or model_config_path == '':
        raise ValueError('Image, metadata, checkpoint and model config paths must be given.')

    device = 'cuda' if args['--cuda'] else 'cpu'

    logging.info('Device: %s', device)
    logging.info('train_fungi_network')

    checkpoint_session = os.path.join(checkpoint_dir, session)
    train_metadata_path = os.path.join(metadata_folder, 'train_metadata.csv')
    test_metadata_path = os.path.join(metadata_folder, 'test_metadata.csv')

    model_config = load_model_config(model_config_path)

    train_fungi_network(train_metadata_path, image_path, checkpoint_session, model_config)
    logging.info('evaluate_network_on_test_set')
    evaluate_network_on_test_set(test_metadata_path, image_path, checkpoint_session, session, model_config)
    
    logging.info('train_fungi_network end')


def evaluate(args:Dict) -> None:
    """
    Evaluate model  
    """   
    # Path to fungi images 
    image_path = args['--image-folder'] if args['--image-folder'] else ''

    # Path to metadata file
    metadata_folder = args['--metadata-folder'] if args['--metadata-folder'] else ''

    # Session name: Change session name for every experiment! 
    # Session name will be saved as the first line of the prediction file
    session = args['--session'] if args['--session'] else "EfficientNet"

    # Folder for results of this experiment based on session name:
    checkpoint_dir = args['--checkpoint-folder'] if args['--checkpoint-folder'] else ''

    model_config_path = args['--model-config'] if args['--model-config'] else ''

    if image_path == '' or metadata_folder == '' or checkpoint_dir == '' or model_config_path == '':
        raise ValueError('Image, metadata, checkpoint and model config paths must be given.')

    device = 'cuda' if args['--cuda'] else 'cpu'

    logging.info('Device: %s', device)

    checkpoint_session = os.path.join(checkpoint_dir, session)
    test_metadata_path = os.path.join(metadata_folder, 'test_metadata.csv')

    model_config = load_model_config(model_config_path)

    logging.info('evaluate_network_on_test_set')
    evaluate_network_on_test_set(
        test_metadata_path, image_path, checkpoint_session, session, model_config)
    
    logging.info('train_fungi_network end')


def main():     
    """ Set logging and call relevant function """
    args = docopt(__doc__)
    
    log_level = args['--log'] if args['--log'] else ''
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=numeric_level)   
    
    
    if args['train']:
        train(args)
    elif args['evaluate']:
        evaluate(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
    