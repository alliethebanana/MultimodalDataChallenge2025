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
    targets_oh = targets_oh[:, :-1]
    
    return targets_oh


def convert_one_hot_to_ints(oh_targets):
    """ Get ints from one-hot encoded """
    num_targets = torch.tensor(oh_targets.shape[1])

    int_targets = []
    for current_oh_target in oh_targets:
        non_zero_idx = torch.nonzero(current_oh_target)
        if len(non_zero_idx) == 0:
            int_targets.append(num_targets)
        else:
            int_targets.append(non_zero_idx[0, 0])
    
    return int_targets
