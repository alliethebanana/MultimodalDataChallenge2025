"""

Util for processing the metadata. E.g. turning the metadata into one-hot encodings 

"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray

import torch

from src.util import get_month_from_date


def get_num_habitats_substrates():
    """
    Get the number of habitats ans substrates
    """
    md_df = pd.read_csv('starting_metadata/test_metadata.csv')

    num_habitats = len(md_df['Habitat'].unique())
    num_substrates = len(md_df['Substrate'].unique())

    return num_habitats, num_substrates 


def translate_habitats_to_class_labels(habitats: pd.Series):
    """ Get class labels from habitat strings """
    possible_habitat_strings = [
        'coniferous woodland/plantation', 'Unmanaged coniferous woodland',
        'Mixed woodland (with coniferous and deciduous trees)',
        'Deciduous woodland', 'Unmanaged deciduous woodland',
        'park/churchyard', 'Acidic oak woodland', 'dune',
        'natural grassland', 'Thorny scrubland', 'lawn', 'hedgerow',
        'garden', 'other habitat', 'meadow', 'roadside', 'Forest bog',
        'wooded meadow, grazing forest', 'Willow scrubland', 'bog',
        'heath', 'gravel or clay pit'] 
    possible_habitat_labels = np.arange(1, len(possible_habitat_strings) + 1)
    habitat_labels = np.zeros(habitats.shape[0])
    nan_habitat_filter = habitats.isna().values
    habitats = habitats.values
    habitats[nan_habitat_filter] = 'nan'
    for i, p_h_str in enumerate(possible_habitat_strings):
        habitat_labels[np.equal(habitats, p_h_str )] = possible_habitat_labels[i]

    return np.array(habitat_labels, dtype=int)


def translate_substrate_to_class_labels(substrates: pd.Series):
    """ Get class labels from habitat strings """
    possible_strings = ['soil', 'dead wood (including bark)',
       'living stems of herbs, grass etc', 'leaf or needle litter',
       'wood and roots of living trees', 'dead stems of herbs, grass etc',
       'bark of living trees', 'stems of herbs, grass etc', 'bark',
       'other substrate', 'wood', 'faeces', 'wood chips or mulch',
       'mosses', 'fungi']
    possible_habitat_labels = np.arange(1, len(possible_strings) + 1)
    labels = np.zeros(substrates.shape[0])
    nan_habitat_filter = substrates.isna().values
    substrates = substrates.values
    substrates[nan_habitat_filter] = 'nan'
    for i, p_h_str in enumerate(possible_strings):
        labels[np.equal(substrates, p_h_str )] = possible_habitat_labels[i]

    return np.array(labels, dtype=int)


def preprocess_dates(dates: NDArray):
    """ Make dates into values """ 
    processed_dates = [0 if pd.isnull(d) else get_month_from_date(d) 
                       for d in dates]

    return np.array(processed_dates, dtype=float)


def preprocess_location(lat: NDArray, long: NDArray):
    """ Preprocess latitude and longitude"""
    prep_lat = lat 
    prep_lat[pd.isnull(lat)] = 0.
    
    prep_long = long
    prep_long[pd.isnull(long)] = 0.

    location = torch.reshape(
        torch.vstack([
            torch.from_numpy(prep_lat), 
            torch.from_numpy(prep_long)]), (-1, 2))
    return location


def preprocess_metadata(metadata_df: pd.DataFrame):
    """ Preprocess metadata to dict with tensors """
    dates = torch.from_numpy(preprocess_dates(metadata_df['eventDate'].values))
    habitats = torch.from_numpy(translate_habitats_to_class_labels(metadata_df['Habitat']))
    substrates = torch.from_numpy(translate_substrate_to_class_labels(metadata_df['Substrate']))
    locations = preprocess_location(
        metadata_df['Latitude'].values, 
        metadata_df['Longitude'].values)
    
    metadata_dict = {
        'eventDate': dates, 
        'Habitat': habitats, 
        'Substrate': substrates, 
        'location': locations}
    
    return metadata_dict
