
import os
import numpy as np
import torch
import torch.nn as nn
from batchrenorm import BatchRenorm2d


class UpNet(nn.Module):

    def __init__(self):
        super(UpNet, self).__init__()
        layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), 
                BatchRenorm2d(64), 
                nn.ReLU()]

        for i in range(15):
            layers.append(nn.Conv2d(64, 64, 3, 1, 1))
            layers.append(BatchRenorm2d(64))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(64, 1, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class DownNet(nn.Module):

    def __init__(self):
        super(DownNet, self).__init__()
        layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), 
                BatchRenorm2d(64), 
                nn.ReLU()]

        for i in range(7):
            layers.append(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, 3, 1, 1))
        layers.append(BatchRenorm2d(64))
        layers.append(nn.ReLU())
        for i in range(6):
            layers.append(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, 3, 1, 1))
        layers.append(BatchRenorm2d(64))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(64, 1, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class BRDNet(nn.Module):

    def __init__(self):
        super(BRDNet, self).__init__()
        self.upnet = UpNet()
        self.dwnet = DownNet()
        self.conv = nn.Conv2d(2, 1, 3, 1, 1)

    def forward(self, x):
        #import pdb;pdb.set_trace()
        out1 = self.upnet(x)
        out2 = self.dwnet(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        out = x - out
        return out




