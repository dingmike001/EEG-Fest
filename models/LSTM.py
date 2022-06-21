import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LSTM(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(LSTM, self).__init__()
        self.channel = cfg.model_parameters.channels
        self.inputsize = cfg.model_parameters.d_model
        self.bi = False
        self.device = device
        self.linear_in = nn.Linear(self.channel, self.inputsize, device=device)

        self.ls1 = nn.LSTM(self.inputsize, hidden_size=self.inputsize, num_layers=1, bias=False, batch_first=True,
                           bidirectional=self.bi, device=device)
        self.re = nn.ReLU()
        self.dropf = nn.Dropout(p=0.1)
        self.dp = nn.Dropout(p=0.2)
        self.ls2 = nn.LSTM(input_size=self.inputsize, hidden_size=self.inputsize, num_layers=1, bias=False, batch_first=True,
                           bidirectional=self.bi, device=device)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.dropf(x)
        x, _ = self.ls1(x)
        x = self.re(x)
        x = self.dp(x)
        x, _ = self.ls2(x)

        return x


class LSTM_Classifier(nn.Module):
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


