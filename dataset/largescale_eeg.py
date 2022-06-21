import torch
from scipy.io import loadmat
import os
import numpy as np
from utils.preprocess import bandpass_filter
class LargeScale_EEG():
    def __init__(self, data_root, mode):
        self.data_root = data_root

        assert (mode == 'train' or mode == 'val'), "mode name is wrong. Currently only support 'train' or 'val'"
        if mode == 'train':
            data = loadmat(os.path.join(data_root, 'large_scale_eeg_train.mat'))
        elif mode == 'val':
            data = loadmat(os.path.join(data_root, 'large_scale_eeg_test.mat'))

        self.samples = data['samples']
        self.labels = data['labels']
        self.sample_size = len(self.samples)
    def __len__(self):
        return(self.sample_size)
    def __getitem__(self, idx):
        samples = self.samples[idx]
        samples = np.transpose(samples)
        use_filter_bank = 'True'
        filter_bank = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]
        if use_filter_bank:
            filter_results = []
            [filter_results.append(bandpass_filter(samples, low=frequency[0], high=frequency[1], fs=200, order=5)) for frequency in filter_bank]
            filter_results = np.stack(filter_results, 0)
        else:
            filter_results = bandpass_filter(samples, low=8, high=30, fs=200, order=5)

        return {'samples': filter_results, 'labels': self.labels[idx]}