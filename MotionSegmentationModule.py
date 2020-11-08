import torch
import torch.nn as nn
from torch.nn import functional as F


class MotionSegmentationModule(nn.Module):
    def __init__(self, input_ch=512, n_channels=2, dim=7):
        # TODO: add the possibility to modfy separately widht and height

        super(MotionSegmentationModule, self).__init__()
        self.input_ch = input_ch
        self.conv1 = nn.Conv2d(in_channels=input_ch,
                               out_channels=100,
                               kernel_size=1,
                               padding=0)
        self.fc1 = nn.Linear(7 * 7 * 100, n_channels * dim * dim)

    def forward(self, x):
        x = F.relu(x)
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # print(x.size())
        # x = torch.reshape(x, (x.size(0), 2 * 7 * 7))  # Dimension 0 is 2 because we are performing a classification task
        return x
