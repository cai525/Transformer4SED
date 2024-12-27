import torch
import torch.nn as nn


class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
        pass

    def forward(self, input):
        #input shape, B,T,C
        return torch.mean(input, dim=1)  #return B,C