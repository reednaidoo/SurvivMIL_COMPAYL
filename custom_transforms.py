import torch
import torch.nn as nn


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0, prob=0.5):
        super(AddGaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.prob = prob

    def forward(self, tensor):
        if self.prob < torch.rand(1):
            device = tensor.device
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise.to(device)
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}, prob={2})".format(
            self.mean, self.std, self.prob
        )