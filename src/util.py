"""

Utility functions

"""


import torch
import torch.nn as nn



def convert_int_targets_to_one_hot(int_targets, num_targets: int):
    """ Get one-hot encoded from ints """
    # pylint: disable=not-callable
    targets_oh = nn.functional.one_hot(
            int_targets, num_targets).float()
    
    return targets_oh


def convert_one_hot_to_ints(oh_targets):
    """ Get ints from one-hot encoded """
    int_targets = []
    for current_oh_target in oh_targets:
        non_zero_idx = torch.nonzero(current_oh_target)
        if len(non_zero_idx) == 0:
            raise ValueError('One-hot vector must have at least one non-zero entry')
        else:
            int_targets.append(non_zero_idx[0, 0])
    
    return torch.tensor(int_targets, dtype=int)


def get_month_from_date(date: str):
    """
    Take only the month from a datestring with format yyyy-MM-DD
    """
    return int(date.split('-')[1])
