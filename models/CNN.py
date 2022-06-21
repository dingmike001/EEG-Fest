import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class cnn_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cnn_block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool_size = (2, 1)
        self.pool_stride = (2, 1)
        self.conv1d1 = nn.Conv1d(self.input_dim, self.output_dim, kernel_size=2, padding=1)
        self.conv1d2 = nn.Conv1d(self.output_dim, self.output_dim, kernel_size=2)
        self.batch = nn.BatchNorm1d(num_features=self.output_dim)

    def forward(self, x):
        x = self.conv1d1(x)
        x = F.relu(x)
        x = self.batch(x)
        x = self.conv1d2(x)
        x = F.relu(x)
        x = self.batch(x)
        return x


class CNN_Classifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.fla = nn.Flatten(start_dim=1)
        self.line = nn.Linear(160, classes)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.sig(x)
        x = self.dp(x)
        x = self.fla(x)
        x = self.line(x)
        return x


class CNNNet(nn.Module):
    def __init__(self, input_dim,d_filt,output_dim):
        super(CNNNet, self).__init__()
        self.block1 = cnn_block(input_dim=input_dim, output_dim=d_filt)
        self.block2 = cnn_block(input_dim=d_filt, output_dim=output_dim)

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.transpose(x, 2, 1)

        return x
