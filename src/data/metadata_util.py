"""

Util for processing the metadata. E.g. turning the metadata into one-hot encodings 

"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from typing import List


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

    return habitat_labels

