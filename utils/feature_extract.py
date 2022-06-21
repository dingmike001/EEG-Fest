import numpy as np
from scipy.stats import differential_entropy as de
from utils.preprocess import bandpass_filter
import torch


#
# def feature_extract(cfg, device, s_datas, s_labels, q_datas, q_labels, model):
#     feature_extract_model = model
#     s_num = len(s_datas)
#     q_num = len(q_datas)
#     filter_bank = cfg.dataset_parameters.filter_parameters.filter_bank_frequencies
#     sample_rate = cfg.dataset_parameters.filter_parameters.sample_rate
#     order = cfg.dataset_parameters.filter_parameters.order
#     shot_num = cfg.train_parameters.shot_num
#     s_sample_labels = s_labels
#     q_sample_labels = q_labels
#
#     for j in range(s_num):
#         s_data = s_datas[j]
#         s_label_data = s_labels[j]
#         for i in range(3):
#             filts = []
#             for k in range(shot_num):
#                 num = i * 3 + k
#                 s_sample = s_data[num]
#                 s_sample_filter = []
#                 [s_sample_filter.append(bandpass_filter(s_sample, low=frequency[0], high=frequency[1],
#                                                         fs=sample_rate, order=order, axis=0)) for frequency in
#                  filter_bank]
#                 s_sample_filter = np.stack(s_sample_filter, 0)
#                 s_sample_filter = de(s_sample_filter, axis=1)
#                 filts.append(s_sample_filter)
#
#             s_sample_filts = np.stack(filts, 0)
#             s_sample_filts = torch.tensor(s_sample_filts).to('cuda').float()
#
#             feature_extract_model.eval()
#             with torch.no_grad():
#                 s_sample_feature = feature_extract_model(s_sample_filts)
#             if i == 0:
#                 group_s_sample_feature = s_sample_feature
#             else:
#                 group_s_sample_feature = torch.cat((group_s_sample_feature, s_sample_feature), dim=0)
#
#         if j == 0:
#             s_sample_features = group_s_sample_feature.unsqueeze(0)
#         else:
#             s_sample_features = torch.cat((s_sample_features,  group_s_sample_feature.unsqueeze(0)), dim=0)
#         print('extract_support_feature: ', j)
#
#     s_sample_labels = torch.tensor(s_sample_labels).to('cuda').long()
#
#     filts = []
#     for j in range(q_num):
#         q_data = q_datas[j]
#         q_sample_filter = []
#         [q_sample_filter.append(bandpass_filter(q_data, low=frequency[0], high=frequency[1],
#                                                 fs=sample_rate, order=order, axis=0)) for frequency in
#          filter_bank]
#         q_sample_filter = np.stack(q_sample_filter, 0)
#         q_sample_filter = de(q_sample_filter, axis=1)
#         # q_sample_filters = np.stack((q_sample_filter, q_sample_filter, q_sample_filter), 0)
#
#         filts.append(q_sample_filter)
#     q_sample_filts = np.stack(filts, 0)
#     q_sample_filters = torch.tensor(q_sample_filts).to('cuda').float()
#
#     feature_extract_model.eval()
#     with torch.no_grad():
#         q_sample_features = feature_extract_model(q_sample_filters)
#
#     print('extract_query_feature')
#     q_sample_labels = torch.tensor(q_sample_labels).to('cuda').long()
#
#     return {'s_features': s_sample_features, 's_labels': s_sample_labels, 'q_features': q_sample_features,
#             'q_labels': q_sample_labels}

#
# def feature_extract(cfg, s_datas, s_labels, q_datas, q_labels, model):
#     feature_extract_model = model
#     filter_bank = cfg.dataset_parameters.filter_parameters.filter_bank_frequencies
#     sample_rate = cfg.dataset_parameters.filter_parameters.sample_rate
#     order = cfg.dataset_parameters.filter_parameters.order
#     shot_num = cfg.train_parameters.shot_num
#     s_sample_labels = []
#     q_sample_labels = q_labels
#     s_datas = s_datas.squeeze(0)
#     q_datas = q_datas.squeeze(0)
#     s_labels = s_labels.squeeze(0)
#
#     for i in range(3):
#         filts = []
#         for k in range(shot_num):
#             num = i * 3 + k
#             s_sample = s_datas[num]
#             s_sample_filter = []
#             [s_sample_filter.append(bandpass_filter(s_sample, low=frequency[0], high=frequency[1], fs=sample_rate,
#                                                     order=order, axis=0)) for frequency in filter_bank]
#             s_sample_filter = np.stack(s_sample_filter, 0)
#             s_sample_filter = de(s_sample_filter, axis=1)
#             filts.append(s_sample_filter)
#             s_sample_label = s_labels[num]
#         s_sample_labels.append(s_sample_label)
#         s_sample_filts = np.stack(filts, 0)
#         s_sample_filts = torch.tensor(s_sample_filts).to('cuda').float()
#         if cfg.train_parameters.zero_grad_feature_trained_model == 'True':
#             feature_extract_model.eval()
#             with torch.no_grad():
#                 s_sample_feature = feature_extract_model(s_sample_filts)
#         else:
#             s_sample_feature = feature_extract_model(s_sample_filts)
#         s_sample_feature = torch.mean(s_sample_feature, dim=0)
#         if i == 0:
#             s_sample_features = s_sample_feature.unsqueeze(0)
#         else:
#             s_sample_features = torch.cat((s_sample_features, s_sample_feature.unsqueeze(0)), dim=0)
#
#     s_sample_labels = torch.tensor(s_sample_labels).to('cuda').long()
#
#
#     q_sample_filter = []
#     [q_sample_filter.append(bandpass_filter(q_datas, low=frequency[0], high=frequency[1],
#                                             fs=sample_rate, order=order, axis=0)) for frequency in filter_bank]
#     q_sample_filter = np.stack(q_sample_filter, 0)
#     q_sample_filter = de(q_sample_filter, axis=1)
#     q_sample_filters = np.stack((q_sample_filter, q_sample_filter, q_sample_filter), 0)
#     q_sample_filters = torch.tensor(q_sample_filters).to('cuda').float()
#     if cfg.train_parameters.zero_grad_feature_trained_model == 'True':
#         feature_extract_model.eval()
#         with torch.no_grad():
#             q_sample_features = feature_extract_model(q_sample_filters)
#     else:
#         q_sample_features = feature_extract_model(q_sample_filters)
#     q_sample_labels = [q_sample_labels,q_sample_labels,q_sample_labels]
#     q_sample_labels = torch.tensor(q_sample_labels).to('cuda').long()
#
#     return {'s_features': s_sample_features, 's_labels': s_sample_labels, 'q_features': q_sample_features,
#             'q_labels': q_sample_labels}


def feature_extract(cfg, s_datas, s_labels, q_datas, q_labels, model):
    feature_extract_model = model
    filter_bank = cfg.dataset_parameters.filter_parameters.filter_bank_frequencies
    sample_rate = cfg.dataset_parameters.filter_parameters.sample_rate
    order = cfg.dataset_parameters.filter_parameters.order
    shot_num = cfg.train_parameters.shot_num
    s_datas_all = s_datas
    q_datas_all = q_datas
    class_num = cfg.model_parameters.classes
    batch_size = s_datas.shape[0]

    for b in range(batch_size):
        s_datas = s_datas_all[b]
        for i in range(class_num):
            filts = []
            for k in range(shot_num):
                num = i * shot_num + k
                s_sample = s_datas[num]
                s_sample_filter = []
                [s_sample_filter.append(bandpass_filter(s_sample, low=frequency[0], high=frequency[1], fs=sample_rate,
                                                        order=order, axis=0)) for frequency in filter_bank]
                s_sample_filter = np.stack(s_sample_filter, 0)
                s_sample_filter = de(s_sample_filter, axis=1)
                filts.append(s_sample_filter)
            s_sample_filts = np.stack(filts, 0)
            s_sample_filts = torch.tensor(s_sample_filts).to('cuda').float()
            s_sample_feature = feature_extract_model(s_sample_filts)
            s_sample_feature = torch.mean(s_sample_feature, dim=0, keepdim=True)
            if i == 0:
                s_sample_features = s_sample_feature
            else:
                s_sample_features = torch.cat((s_sample_features, s_sample_feature), dim=0)
        if b == 0:
            s_sample_features_batch = s_sample_features.unsqueeze(0)
        else:
            s_sample_features_batch = torch.cat((s_sample_features_batch, s_sample_features.unsqueeze(0)), dim=0)

    for b in range(batch_size):
        q_datas = q_datas_all[b]
        q_sample_filter = []
        [q_sample_filter.append(bandpass_filter(q_datas, low=frequency[0], high=frequency[1],
                                                fs=sample_rate, order=order, axis=0)) for frequency in filter_bank]
        q_sample_filter = np.stack(q_sample_filter, 0)
        q_sample_filter = de(q_sample_filter, axis=1)
        q_sample_filter = np.expand_dims(q_sample_filter, axis=0)
        q_sample_filters = np.repeat(q_sample_filter, class_num, axis=0)
        q_sample_filters = torch.tensor(q_sample_filters).to('cuda').float()
        q_sample_features = feature_extract_model(q_sample_filters)
        if b == 0:
            q_sample_features_batch = q_sample_features.unsqueeze(0)
        else:
            q_sample_features_batch = torch.cat((q_sample_features_batch, q_sample_features.unsqueeze(0)), dim=0)
    return {'s_features': s_sample_features_batch,  'q_features': q_sample_features_batch}



def feature_extract_4_class(cfg, s_datas, s_labels, q_datas, q_labels, model):
    feature_extract_model = model
    filter_bank = cfg.dataset_parameters.filter_parameters.filter_bank_frequencies
    sample_rate = cfg.dataset_parameters.filter_parameters.sample_rate
    order = cfg.dataset_parameters.filter_parameters.order
    shot_num = cfg.train_parameters.shot_num
    s_datas_all = s_datas
    q_datas_all = q_datas
    class_num = cfg.model_parameters.output_classes
    batch_size = s_datas.shape[0]

    for b in range(batch_size):
        s_datas = s_datas_all[b]
        for i in range(class_num):
            filts = []
            for k in range(shot_num):
                num = i * shot_num + k
                s_sample = s_datas[num]
                s_sample_filter = []
                [s_sample_filter.append(bandpass_filter(s_sample, low=frequency[0], high=frequency[1], fs=sample_rate,
                                                        order=order, axis=0)) for frequency in filter_bank]
                s_sample_filter = np.stack(s_sample_filter, 0)
                s_sample_filter = de(s_sample_filter, axis=1)
                filts.append(s_sample_filter)
            s_sample_filts = np.stack(filts, 0)
            s_sample_filts = torch.tensor(s_sample_filts).to('cuda').float()
            s_sample_feature = feature_extract_model(s_sample_filts)
            s_sample_feature = torch.mean(s_sample_feature, dim=0, keepdim=True)
            if i == 0:
                s_sample_features = s_sample_feature
            else:
                s_sample_features = torch.cat((s_sample_features, s_sample_feature), dim=0)
        if b == 0:
            s_sample_features_batch = s_sample_features.unsqueeze(0)
        else:
            s_sample_features_batch = torch.cat((s_sample_features_batch, s_sample_features.unsqueeze(0)), dim=0)

    for b in range(batch_size):
        q_datas = q_datas_all[b]
        q_sample_filter = []
        [q_sample_filter.append(bandpass_filter(q_datas, low=frequency[0], high=frequency[1],
                                                fs=sample_rate, order=order, axis=0)) for frequency in filter_bank]
        q_sample_filter = np.stack(q_sample_filter, 0)
        q_sample_filter = de(q_sample_filter, axis=1)
        q_sample_filter = np.expand_dims(q_sample_filter, axis=0)
        q_sample_filters = np.repeat(q_sample_filter, class_num, axis=0)
        q_sample_filters = torch.tensor(q_sample_filters).to('cuda').float()
        q_sample_features = feature_extract_model(q_sample_filters)
        if b == 0:
            q_sample_features_batch = q_sample_features.unsqueeze(0)
        else:
            q_sample_features_batch = torch.cat((q_sample_features_batch, q_sample_features.unsqueeze(0)), dim=0)
    return {'s_features': s_sample_features_batch,  'q_features': q_sample_features_batch}

