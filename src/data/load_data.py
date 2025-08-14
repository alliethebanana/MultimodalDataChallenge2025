"""

For loading data

"""


import os

from sklearn.model_selection import train_test_split

import pandas as pd

from torch.utils.data import DataLoader, Dataset

from albumentations import Compose, Normalize, Resize
from albumentations import RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

from src.data.metadata_util import preprocess_metadata


def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
    width, height = 224, 224
    if data == 'train':
        return Compose([
            RandomResizedCrop((width, height), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("Unknown data mode requested (only 'train' or 'valid' allowed).")


class FungiDataset(Dataset):
    """
    Making the fungi dataset
    """
    def __init__(self, df, path, train_test_final: str, use_dino: bool, transform=None):
        self.cluster_index = pd.read_csv('cluster_index.csv')
        if train_test_final == 'train':
            ban_files = self.cluster_index[self.cluster_index['handdrawn'].values]['filename_index']
            df = df[~df['filename_index'].isin(ban_files)]

        self.df = df
        self.metadata_dict = preprocess_metadata(df)
        self.transform = transform
        self.path = path
        self.train_val_test = train_test_final
        self.use_dino = use_dino

        
        self.dino_order = pd.read_csv(f'{train_test_final}_order.csv').values

        dino_features_path = f'image_features/dino_features_resize_1302_{train_test_final}.npy'
        self.dino_features_array = np.load(dino_features_path)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['filename_index'].values[idx]
        # Get label if it exists; otherwise return None
        label = self.df['taxonID_index'].values[idx]  # Get label
        if pd.isnull(label):
            label = -1  # Handle missing labels for the test dataset
        else:
            label = int(label)
        
        if self.use_dino:
            dino_filter = (self.dino_order == file_path).squeeze()
            image = self.dino_features_array[dino_filter].squeeze()
        else:
            with Image.open(os.path.join(self.path, file_path)) as img:
                # Convert to RGB mode (handles grayscale images as well)
                image = img.convert('RGB')
            image = np.array(image)

            # Apply transformations if available
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
        date = self.metadata_dict['eventDate'][idx]
        habitat = self.metadata_dict['Habitat'][idx]
        substrate = self.metadata_dict['Substrate'][idx]
        location = self.metadata_dict['location'][idx]

        md = (date, habitat, substrate, location)

        return image, label, file_path, md


def get_train_dataloaders(
        metadata_path: str, image_path: str, 
        use_dino: bool, num_workers: int = 2):
    """ Get dataloaders for training and validation """
    # Load metadata
    df = pd.read_csv(metadata_path)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print('Training size', len(train_df))
    print('Validation size', len(val_df))
    
    # Initialize DataLoaders
    transform = None if use_dino else get_transforms(data='valid')
    train_dataset = FungiDataset(train_df, image_path, 'train', use_dino, transform=transform)
    valid_dataset = FungiDataset(val_df, image_path, 'train', use_dino, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader


def get_test_dataloader(
        metadata_path: str, image_path: str,
        use_dino: bool, num_workers: int = 2):
    """ Get dataloader for test """
    df = pd.read_csv(metadata_path)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    transform = None if use_dino else get_transforms(data='valid')
    test_dataset = FungiDataset(
        test_df, image_path, 'test', use_dino, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    return test_loader


def get_final_dataloader(
        metadata_path: str, image_path: str,
        use_dino: bool, num_workers: int = 2):
    """ Get dataloader for test """
    df = pd.read_csv(metadata_path)
    final_df = df[df['filename_index'].str.startswith('fungi_final')]
    transform = None if use_dino else get_transforms(data='valid')
    test_dataset = FungiDataset(
        final_df, image_path, 'final', use_dino, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    return test_loader
