"""

Util for processing the metadata. E.g. turning the metadata into one-hot encodings 

"""

import pandas as pd


def get_num_habitats_substrates():
    """
    Get the number of habitats ans substrates
    """
    md_df = pd.read_csv('starting_metadata/test_metadata.csv')

    num_habitats = len(md_df['Habitat'].unique())
    num_substrates = len(md_df['Substrate'].unique())

    return num_habitats, num_substrates 


