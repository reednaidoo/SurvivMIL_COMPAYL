import torch
import torch.nn as nn
from custom_transforms import AddGaussianNoise


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, mean=0.0, std=0.1, prob=0.5) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            AddGaussianNoise(mean, std, prob),
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_out = self.transforms(x)

        return x_out