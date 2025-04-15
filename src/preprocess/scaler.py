import torch
from torch import nn 

class Scaler(nn.Module):
    def __init__(self, statistic="instance", normtype="minmax", dims=(0, 2), eps=1e-8):
        super(Scaler, self).__init__()
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == "dataset":
            super(Scaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.statistic == "dataset":
            super(Scaler, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                      unexpected_keys, error_msgs)

    def forward(self, input):
        if self.statistic == "dataset":
            if self.normtype == "mean":
                return input - self.mean
            elif self.normtype == "standard":
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (input - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError

        elif self.statistic =="instance":
            if self.normtype == "mean":
                return input - torch.mean(input, self.dims, keepdim=True)
            elif self.normtype == "standard":
                return (input - torch.mean(input, self.dims, keepdim=True)) / (
                        torch.std(input, self.dims, keepdim=True) + self.eps)
            elif self.normtype == "minmax":
                return (input - torch.amin(input, dim=self.dims, keepdim=True)) / (
                    torch.amax(input, dim=self.dims, keepdim=True)
                    - torch.amin(input, dim=self.dims, keepdim=True) + self.eps)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError