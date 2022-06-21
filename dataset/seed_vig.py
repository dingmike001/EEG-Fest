import torch
from scipy.stats import differential_entropy as de
import mat73
import os
import numpy as np
from utils.preprocess import bandpass_filter
from scipy.stats import differential_entropy as de


def load_data_and_filt(data_path):
    data = mat73.loadmat(data_path)
    samples = data['samples']
    labels = data['labels']
    return samples, labels


class SEED_VIG():
    def __init__(self, cfg, samples, labels):
        self.samples = samples
        self.labels = labels
        self.sample_size = len(samples)
        self.use_filter_bank = cfg.dataset_parameters.use_filter_bank
        if self.use_filter_bank:
            self.band_frequency = cfg.dataset_parameters.filter_parameters.filter_bank_frequencies
            self.sample_rate = cfg.dataset_parameters.filter_parameters.sample_rate
            self.order = cfg.dataset_parameters.filter_parameters.order
        else:
            self.band_frequency = cfg.dataset_parameters.filter_parameters.filter_bandwidth
            self.sample_rate = cfg.dataset_parameters.filter_parameters.sample_rate
            self.order = cfg.dataset_parameters.filter_parameters.order

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        samples = self.samples[idx]
        samples = np.transpose(samples)
        if self.use_filter_bank:
            filter_bank = self.band_frequency
            filter_results = []
            [filter_results.append(bandpass_filter(samples, low=frequency[0], high=frequency[1],
                                                   fs=self.sample_rate, order=self.order)) for frequency in filter_bank]
            filter_results = np.stack(filter_results, 0)
            filter_results = de(filter_results, axis=2)
        else:
            filter_results = bandpass_filter(samples, low=self.band_frequency[0], high=self.band_frequency[1],
                                             fs=self.sample_rate, order=self.order)

        return {'samples': filter_results, 'labels': self.labels[idx]}


class SEED_VIG_few_shot():
    def __init__(self, s_samples, s_labels, q_samples, q_labels):
        self.s_samples = s_samples
        self.s_labels = s_labels
        self.q_samples = q_samples
        self.q_labels = q_labels
        self.sample_size = len(self.s_samples)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        s_sample = self.s_samples[idx]
        s_label = self.s_labels[idx]
        q_sample = self.q_samples[idx]
        q_label = self.q_labels[idx]

        return {'s_sample': s_sample, 'q_sample': q_sample, 's_target': s_label, 'q_target': q_label}


class SEED_VIG_few_shot_binary():
    def __init__(self, s_samples, s_labels, q_samples, q_labels):
        self.s_samples = s_samples
        self.s_labels = s_labels
        self.q_samples = q_samples
        self.q_labels = q_labels
        self.sample_size = len(self.s_samples)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        s_sample = self.s_samples[idx]
        s_label = self.s_labels[idx]
        q_sample = self.q_samples[idx]
        q_label = self.q_labels[idx]

        return {'s_sample': s_sample, 'q_sample': q_sample, 's_target': s_label, 'q_target': q_label}


class movie_ix():
    def __init__(self, samples):
        self.samples = samples
        self.sample_size = len(self.samples)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        q_sample = self.samples[idx]
        return {'q_sample': q_sample}
