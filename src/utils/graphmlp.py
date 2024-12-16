import torch
import torch.nn as nn
import torch.nn.functional as F


class GMLP(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout, output_dim=64, num_fc_layers=3, act_fn='gelu'):
        super(GMLP, self).__init__()
        self.fc_layers = nn.ModuleList()
        for i in range(num_fc_layers):
            in_features = input_dim if i == 0 else hid_dim
            out_features = hid_dim if i < num_fc_layers - 1 else output_dim
            self.fc_layers.append(nn.Linear(in_features, out_features))
        
        # Dictionary of activation functions
        activation_functions = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'gelu': F.gelu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'elu': F.elu,
            'selu': F.selu,
            'softplus': F.softplus
        }
        
        # Set the activation function
        if act_fn in activation_functions:
            self.act_fn = activation_functions[act_fn]
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim, eps=1e-6)
        
        self._init_weights()

    def _init_weights(self):
        for fc in self.fc_layers:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.normal_(fc.bias, std=1e-6)

    def forward(self, x):
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:
                x = self.act_fn(x)
                x = self.layernorm(x)
                x = self.dropout(x)
        return x