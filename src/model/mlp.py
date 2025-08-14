"""

A multilayer perceptron model

"""

import torch.nn as nn



class SmallMLP(nn.Module):
    """ A small MLP network
    """
    def __init__(self, input_dim:int, final_dim:int, 
    num_features:int = 64, nonlinearity: str = 'leaky_relu'):
        super().__init__()
        
        self.nonlinearity = nonlinearity
        self.final_dim = final_dim
        self.num_features = num_features        
        
        # Layers
        self.l1 = nn.Linear(in_features=input_dim, out_features=self.num_features, bias=True)
        self.l2 = nn.Linear(
            in_features=self.num_features, out_features=self.num_features, bias=True)
        self.l3 = nn.Linear(
            in_features=self.num_features, out_features=self.num_features, bias=True)
        self.l4 = nn.Linear(
            in_features=self.num_features, out_features=self.final_dim, bias=False)
                
        if self.nonlinearity == 'relu':
            self.activation = nn.ReLU()
        elif self.nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f'Nonlinearity not implemented: {self.nonlinearity}')
        
    
    def forward(self, x):
        """ The forward pass """
        out = self.l1(x)
        out = self.activation(out)
        out = self.l2(out)
        out = self.activation(out)
        out = self.l3(out)
        out = self.activation(out)
        out = self.l4(out)
        return out
