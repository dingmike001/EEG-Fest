from scipy.io import loadmat
from scipy.signal import butter, filtfilt
# from mne.filter import filter_data
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import mat73
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from scipy.stats import differential_entropy as de


def dataset_ss(train_data_dir, val_data_dir):
    train_data, train_label = load_data_and_nofilt(train_data_dir)
    val_data, val_label = load_data_and_nofilt(val_data_dir)
    dataset_train = SUS_TAIN(samples=train_data, labels=train_label)
    dataset_val = SUS_TAIN(samples=val_data, labels=val_label)
    data_train = DataLoader(dataset_train, batch_size=32, shuffle=False)
    data_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
    return data_train, data_val


class SUS_TAIN():
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.sample_size = len(samples)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        samples = self.samples[idx]
        labels = self.labels[idx]

        return {'eeg': samples, 'label': labels}


def load_data_and_nofilt(data_path):
    data = mat73.loadmat(data_path)
    samples = data['samples']
    labels = data['labels']
    return samples, labels


def bandpass_filter(data, low, high, fs, order=5, axis=1):
    b, a = butter(order, [low, high], fs=fs, btype='bandpass', analog=False)
    y = filtfilt(b, a, data, axis=axis)
    return y


def load_data_and_filt(data_path):
    data = mat73.loadmat(data_path)
    samples = data['samples']
    labels = data['labels']
    filter_bank = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]
    filter_results = []
    aa = np.ones([len(labels)]) * 6.8
    labels = np.minimum(labels, aa)
    [filter_results.append(bandpass_filter(samples, low=frequency[0], high=frequency[1],
                                           fs=200, order=5)) for frequency in filter_bank]
    filter_results = np.stack(filter_results, 1)
    filter_results = de(filter_results, axis=2)

    return filter_results, labels


class SEED_VIG_load():
    def __init__(self, eeg_root, label_root, random_state=0, train_percentage=0.75, mode='train'):
        self.eeg_root = eeg_root
        self.label_root = label_root
        self.random_state = random_state
        self.train_percentage = train_percentage
        self.mode = mode
        self.total_eeg = np.transpose(loadmat(self.eeg_root)['de_movingAve'], (1, 0, 2))
        # self.total_eeg = loadmat(self.eeg_root)['a_modified']
        # self.total_eeg = loadmat(self.eeg_root)['data']
        # self.total_eeg_filter = filter_data(self.total_eeg, sfreq=200, l_freq=0.15, h_freq=45, verbose=False)
        # self.total_labels = loadmat(self.label_root)['label_modified']
        # self.total_labels = loadmat(self.label_root)['label']
        self.total_labels = loadmat(self.label_root)['perclos']
        self.eeg, self.labels = self.train_val_split()

    def __len__(self):
        return len(self.labels)

    def train_val_split(self):
        self.train_data, self.val_data, self.train_label, self.val_label = train_test_split(self.total_eeg,
                                                                                            self.total_labels,
                                                                                            test_size=1 - self.train_percentage,
                                                                                            train_size=self.train_percentage,
                                                                                            random_state=self.random_state)

        if self.mode == 'train':
            return self.train_data, self.train_label
        return self.val_data, self.val_label

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        label = self.labels[idx]
        eeg_sample = {'eeg': eeg, 'label': label}
        return eeg_sample


def loaddata(testdata_path, labeldata_path):
    dataset_train = SEED_VIG_load(testdata_path, labeldata_path, random_state=0, train_percentage=0.75, mode='train')
    dataset_val = SEED_VIG_load(testdata_path, labeldata_path, random_state=0, train_percentage=0.75, mode='val')
    data_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    data_val = DataLoader(dataset_val, batch_size=1, shuffle=True)
    return data_train, data_val


def loaddatawithtest(testdata_path, labeldata_path):
    dataset_train = SEED_VIG_load(testdata_path, labeldata_path, random_state=0, train_percentage=0.8, mode='train')
    dataset_val = SEED_VIG_load(testdata_path, labeldata_path, random_state=0, train_percentage=0.8, mode='val')
    dataset_test = SEED_VIG_load(testdata_path, labeldata_path, random_state=0, train_percentage=0.6, mode='train')
    data_train = DataLoader(dataset_train, batch_size=16, shuffle=False)
    data_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
    data_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    return data_train, data_val, data_test


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(data):
    """Data Loader"""
    data_dir = os.path.join(data)

    data = pd.read_csv(data_dir,
                       # infer_datetime_format=True,
                       parse_dates=['date']
                       )

    data.index = data['date']
    data = data.drop('date', axis=1)

    return data


def plot_full(path, data, feature):
    """Plot Full Graph of Energy Dataset"""
    data.plot(y=feature, figsize=(16, 8))
    plt.xlabel('DateTime', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylabel(feature, fontsize=10)
    plt.grid()
    plt.title('{} Energy Prediction'.format(feature))
    plt.savefig(os.path.join(path, '{} Energy Prediction.png'.format(feature)))
    plt.show()


def split_sequence_uni_step(sequence, n_steps):
    """Rolling Window Function for Uni-step"""
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def split_sequence_multi_step(sequence, n_steps_in, n_steps_out):
    """Rolling Window Function for Multi-step"""
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(sequence):
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)[:, :, 0]


def data_loader(x, y, train_split, test_split, batch_size):
    """Prepare data by applying sliding windows and return data loader"""

    # Split to Train, Validation and Test Set #
    train_seq, test_seq, train_label, test_label = train_test_split(x, y, train_size=train_split, shuffle=False)
    val_seq, test_seq, val_label, test_label = train_test_split(test_seq, test_label, train_size=test_split,
                                                                shuffle=False)

    # Convert to Tensor #
    train_set = TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_label))
    val_set = TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(val_label))
    test_set = TensorDataset(torch.from_numpy(test_seq), torch.from_numpy(test_label))

    # Data Loader #
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_lr_scheduler(lr_scheduler, optimizer):
    """Learning Rate Scheduler"""
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01,
                                                               patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler


def percentage_error(actual, predicted):
    """Percentage Error"""
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_percentage_error(y_true, y_pred):
    """Mean Percentage Error"""
    mpe = np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100
    return mpe


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mape = np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
    return mape


def plot_pred_test(pred, actual, path, feature, model, step):
    """Plot Test set Prediction"""
    plt.figure(figsize=(10, 8))

    plt.plot(pred, label='Pred')
    plt.plot(actual, label='Actual')

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('{}'.format(feature), fontsize=18)

    plt.legend(loc='best')
    plt.grid()

    plt.title('{} Energy Prediction using {} and {}'.format(feature, model.__class__.__name__, step), fontsize=18)
    plt.savefig(
        os.path.join(path, '{} Energy Prediction using {} and {}.png'.format(feature, model.__class__.__name__, step)))
