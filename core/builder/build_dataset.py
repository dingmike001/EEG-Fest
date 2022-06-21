from torch.utils.data import DataLoader
from dataset.seed_vig import load_data_and_filt
from dataset.seed_vig import SEED_VIG, SEED_VIG_few_shot, SEED_VIG_few_shot_binary, movie_ix
import mat73
from scipy.io import loadmat
from utils.feature_extract import feature_extract
import torch


def dataset_builder(cfg):
    train_data, train_label = load_data_and_filt(cfg.dataset_parameters.train_data_dir)
    val_data, val_label = load_data_and_filt(cfg.dataset_parameters.val_data_dir)
    dataset_train = SEED_VIG(cfg=cfg, samples=train_data, labels=train_label)
    dataset_val = SEED_VIG(cfg=cfg, samples=val_data, labels=val_label)
    data_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    data_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
    return data_train, data_val


def val_dataset_bulder(cfg):
    val_data, val_label = load_data_and_filt(cfg.dataset_parameters.val_data_dir)
    dataset_val = SEED_VIG(cfg=cfg, samples=val_data, labels=val_label)
    data_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
    return data_val


def few_shot_dataset_builder(cfg, device):
    data = mat73.loadmat(cfg.dataset_parameters.test_data_dir)
    s_samples = data['s_samples']
    s_labels = data['s_labels']
    q_samples = data['q_samples']
    q_labels = data['q_labels']

    s_q_dataset = SEED_VIG_few_shot(s_samples=s_samples, s_labels=s_labels,
                                    q_samples=q_samples, q_labels=q_labels)
    test_data_loader = DataLoader(dataset=s_q_dataset, batch_size=1,
                                  num_workers=cfg.dataset_parameters.num_workers, shuffle=False)

    data = mat73.loadmat(cfg.dataset_parameters.val_data_dir)
    s_samples1 = data['s_samples']
    s_labels1 = data['s_labels']
    q_samples1 = data['q_samples']
    q_labels1 = data['q_labels']

    s_q_dataset1 = SEED_VIG_few_shot(s_samples=s_samples1, s_labels=s_labels1,
                                     q_samples=q_samples1, q_labels=q_labels1)
    val_data_loader = DataLoader(dataset=s_q_dataset1, batch_size=1,
                                 num_workers=cfg.dataset_parameters.num_workers, shuffle=False)

    return test_data_loader, val_data_loader


def few_shot_dataset_binary_builder(cfg, device):
    data = mat73.loadmat(cfg.dataset_parameters.test_data_dir)
    s_samples = data['s_samples']
    s_labels = data['s_labels']
    q_samples = data['q_samples']
    q_labels = data['q_labels']

    s_q_dataset = SEED_VIG_few_shot_binary(s_samples=s_samples, s_labels=s_labels,
                                           q_samples=q_samples, q_labels=q_labels)
    test_data_loader = DataLoader(dataset=s_q_dataset, batch_size=cfg.dataset_parameters.train_batch_size,
                                  num_workers=cfg.dataset_parameters.num_workers, shuffle=False)

    data = mat73.loadmat(cfg.dataset_parameters.val_data_dir)
    s_samples1 = data['s_samples']
    s_labels1 = data['s_labels']
    q_samples1 = data['q_samples']
    q_labels1 = data['q_labels']

    s_q_dataset1 = SEED_VIG_few_shot_binary(s_samples=s_samples1, s_labels=s_labels1,
                                            q_samples=q_samples1, q_labels=q_labels1)
    val_data_loader = DataLoader(dataset=s_q_dataset1, batch_size=cfg.dataset_parameters.val_batch_size,
                                 num_workers=cfg.dataset_parameters.num_workers, shuffle=False)

    return test_data_loader, val_data_loader


def movie_builder(cfg, device):
    data = mat73.loadmat(cfg.dataset_parameters.movie_data_dir)
    q_samples = data['seed_movie_all_random']

    movie_dataset = movie_ix(samples=q_samples)
    movie_dataloade = DataLoader(dataset=movie_dataset, batch_size=1, shuffle=False)
    return movie_dataloade
