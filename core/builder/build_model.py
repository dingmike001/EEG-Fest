from models.self_attention import AttentiontTransformer
from models.self_attention_ning import AttentionTransformer_Ning, Classifier
from models.cross_attention import CrossAttention_feature, CrossAttention_feature_class
from models.LSTM import LSTM, LSTM_Classifier
from models.CNN import CNNNet, CNN_Classifier
import torch.nn as nn
import torch.nn.functional as F
from utils.feature_extract import feature_extract, feature_extract_4_class
import torch


def get_self_attention_parameter(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    trained_dict = model.state_dict()
    deletekey = ['classifier.ln2.weight', 'classifier.ln2.bias', 'classifier.ln1.weight', 'classifier.ln1.bias']
    new = {k: v for k, v in trained_dict.items() if k not in deletekey}
    return new


def get_feature_extractor_model(cfg, device):
    model = EEG_Classification(cfg, device)
    model = model.to(device)
    get_p = get_self_attention_parameter(model, path=cfg.model_parameters.pre_trained_self_attention_save_path,
                                         device=device)
    model = sfa_feature_extractor(cfg, device)
    model.load_state_dict(get_p)
    model.to(device)
    return model


def get_lstm_parameter(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    trained_dict = model.state_dict()
    deletekey = ['classifier.line.weight', 'classifier.line.bias']
    new = {k: v for k, v in trained_dict.items() if k not in deletekey}
    return new


def get_LSTM_feature_extractor_model(cfg, device):
    model = EEG_Classification(cfg, device)
    model = model.to(device)
    get_p = get_lstm_parameter(model, path=cfg.model_parameters.pre_trained_lstm_save_path,
                               device=device)
    model = lstm_feature_extractor(cfg, device)
    model.load_state_dict(get_p)
    model.to(device)
    return model


def get_cnn_parameter(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    trained_dict = model.state_dict()
    deletekey = ['classifier.line.weight', 'classifier.line.bias']
    new = {k: v for k, v in trained_dict.items() if k not in deletekey}
    return new


def get_cnn_feature_extractor_model(cfg, device):
    model = EEG_Classification(cfg, device)
    model = model.to(device)
    get_p = get_cnn_parameter(model, path=cfg.model_parameters.pre_trained_cnn_save_path,
                              device=device)
    model = cnn_feature_extractor(cfg, device)
    model.load_state_dict(get_p)
    model.to(device)
    return model


class EEG_Classification(nn.Module):
    def __init__(self, cfg, device):
        super(EEG_Classification, self).__init__()
        if cfg.model_parameters.model_name == 'ce-zhang':
            max_len = int((cfg.model_parameters.seq_len - 1) // 2) * cfg.model_parameters.num_heads
            self.feature_extractor = AttentiontTransformer(eeg_channel=cfg.model_parameters.channels,
                                                           d_model=cfg.model_parameters.d_model,
                                                           n_head=cfg.model_parameters.num_heads,
                                                           d_hid=cfg.model_parameters.d_hidden,
                                                           n_layers=cfg.model_parameters.n_layers,
                                                           dropout=cfg.model_parameters.dropout,
                                                           max_len=max_len, device=device)
            self.classifier = Classifier(40, cfg.model_parameters.classes,
                                         cfg.model_parameters.dropout,
                                         cfg.model_parameters.d_model)
        elif cfg.model_parameters.model_name == 'ning-ding':
            self.feature_extractor = AttentionTransformer_Ning(eeg_channel=cfg.model_parameters.channels,
                                                               d_model=cfg.model_parameters.d_model,
                                                               n_head=cfg.model_parameters.num_heads,
                                                               d_hid=cfg.model_parameters.d_hidden,
                                                               n_layers=cfg.model_parameters.n_layers,
                                                               dropout=cfg.model_parameters.dropout,
                                                               seq_len=cfg.model_parameters.seq_len,
                                                               use_model_selection=cfg.model_parameters.model_selection,
                                                               device=device)
            self.classifier = Classifier(40, cfg.model_parameters.classes,
                                         cfg.model_parameters.dropout,
                                         cfg.model_parameters.d_model)
        elif cfg.model_parameters.model_name == 'LSTM':
            self.feature_extractor = LSTM(cfg, device=device)
            self.classifier = LSTM_Classifier(classes=cfg.model_parameters.classes)
        elif cfg.model_parameters.model_name == 'CNN':
            self.feature_extractor = CNNNet(input_dim=cfg.model_parameters.channels, d_filt=cfg.model_parameters.d_filt,
                                            output_dim=cfg.model_parameters.d_model)
            self.classifier = CNN_Classifier(classes=cfg.model_parameters.classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class conv_block(nn.Module):
    def __init__(self):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 2)
        self.conv2 = nn.Conv2d(1, 1, 2)
        self.conv3 = nn.Conv2d(1, 1, 2)
        self.conv4 = nn.Conv2d(1, 1, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class cnn_feature_extractor(nn.Module):
    def __init__(self, cfg, device):
        super(cnn_feature_extractor, self).__init__()
        self.feature_extractor = CNNNet(input_dim=cfg.model_parameters.channels, d_filt=cfg.model_parameters.d_filt,
                                        output_dim=cfg.model_parameters.d_model)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class sfa_feature_extractor(nn.Module):
    def __init__(self, cfg, device):
        super(sfa_feature_extractor, self).__init__()
        self.feature_extractor = AttentionTransformer_Ning(eeg_channel=cfg.model_parameters.channels,
                                                           d_model=cfg.model_parameters.d_model,
                                                           n_head=cfg.model_parameters.num_heads,
                                                           d_hid=cfg.model_parameters.d_hidden,
                                                           n_layers=cfg.model_parameters.n_layers,
                                                           dropout=cfg.model_parameters.dropout,
                                                           seq_len=cfg.model_parameters.seq_len,
                                                           use_model_selection=cfg.model_parameters.model_selection,
                                                           device=device)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class lstm_feature_extractor(nn.Module):
    def __init__(self, cfg, device):
        super(lstm_feature_extractor, self).__init__()
        self.feature_extractor = LSTM(cfg, device=device)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class CrossAttention_Distance(nn.Module):
    def __init__(self, cfg, device):
        super(CrossAttention_Distance, self).__init__()

        self.cross_attention = CrossAttention_feature(d_model=cfg.model_parameters.d_model,
                                                      n_head=cfg.model_parameters.num_heads,
                                                      dropout=cfg.model_parameters.dropout,
                                                      device=device)

        if cfg.model_parameters.model_name == 'LSTM':
            self.feature_extraction_model = get_LSTM_feature_extractor_model(cfg, device).requires_grad_(False)
        elif cfg.model_parameters.model_name == 'CNN':
            self.feature_extraction_model = get_cnn_feature_extractor_model(cfg, device).requires_grad_(False)
        else:
            self.feature_extraction_model = get_feature_extractor_model(cfg, device).requires_grad_(False)
        self.cfg = cfg
        self.device = device
        self.max2d1 = nn.MaxPool2d((5, 1))
        self.max2d2 = nn.MaxPool2d((5, 1))
        self.conv_block1 = conv_block()
        self.conv_block2 = conv_block()

        self.avg2d1 = nn.AvgPool2d((5, 1))
        self.avg2d2 = nn.AvgPool2d((5, 1))
        self.lin1 = nn.Linear(3, 4)
        self.lin2 = nn.Linear(3, 4)
        self.norm1 = nn.LayerNorm(cfg.model_parameters.d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(cfg.model_parameters.d_model, eps=1e-5)

    def forward(self, s_samples, q_samples, s_labels, q_labels):
        features_labels = feature_extract(cfg=self.cfg, q_datas=q_samples, s_datas=s_samples,
                                          s_labels=s_labels, q_labels=q_labels, model=self.feature_extraction_model)
        support_features = features_labels['s_features']
        query_features = features_labels['q_features']
        cam_features = self.cross_attention(support_features, query_features)
        s_cam_f = cam_features['s_ca_f']
        q_cam_f = cam_features['q_ca_f']
        # s_cam_f = self.max2d1(s_cam_f)
        # q_cam_f = self.max2d2(q_cam_f)
        s_cam_f = self.avg2d1(s_cam_f)
        q_cam_f = self.avg2d2(q_cam_f)

        # s_cam_f=s_cam_f.unsqueeze(1)
        # q_cam_f=q_cam_f.unsqueeze(1)

        # s_cam_f = self.conv_block1(s_cam_f)
        # q_cam_f = self.conv_block2(q_cam_f)
        #
        s_cam_f = s_cam_f.squeeze(2)
        q_cam_f = q_cam_f.squeeze(2)
        # s_cam_f = torch.transpose(s_cam_f, 2, 1)
        # q_cam_f = torch.transpose(q_cam_f, 2, 1)
        # s_cam_f = self.lin1(s_cam_f)
        # q_cam_f = self.lin2(q_cam_f)
        # s_cam_f = torch.transpose(s_cam_f, 2, 1)
        # q_cam_f = torch.transpose(q_cam_f, 2, 1)
        # s_cam_f = self.norm1(s_cam_f)
        # q_cam_f = self.norm2(q_cam_f)
        score = F.pairwise_distance(s_cam_f, q_cam_f, p=2)
        # score = torch.reshape(score, (self.cfg.model_parameters.classes,self.cfg.train_parameters.shot_num))
        # score = torch.mean(score, dim=-1)
        score = -score

        # support_features = self.max2d1(support_features)
        # query_features = self.max2d2(query_features)
        #
        # support_features = support_features.squeeze()
        # query_features = query_features.squeeze()
        # score = F.pairwise_distance(support_features, query_features, p=2)
        # score = -score

        return score


class CrossAttention_flatten(nn.Module):
    def __init__(self, cfg, device):
        super(CrossAttention_flatten, self).__init__()

        self.cross_attention = CrossAttention_feature(d_model=cfg.model_parameters.d_model,
                                                      n_head=cfg.model_parameters.num_heads,
                                                      dropout=cfg.model_parameters.dropout,
                                                      device=device)
        if cfg.model_parameters.model_name == 'LSTM':
            self.feature_extraction_model = get_LSTM_feature_extractor_model(cfg, device).requires_grad_(False)
        elif cfg.model_parameters.model_name == 'CNN':
            self.feature_extraction_model = get_cnn_feature_extractor_model(cfg, device).requires_grad_(False)
        else:
            self.feature_extraction_model = get_feature_extractor_model(cfg, device).requires_grad_(False)
        self.cfg = cfg
        self.device = device
        self.avg2d1 = nn.AvgPool2d((5, 1))
        self.avg2d2 = nn.AvgPool2d((5, 1))
        self.batch = nn.BatchNorm1d(num_features=cfg.model_parameters.d_model)
        # self.lin = nn.Linear(64, 2)
        self.lin = nn.Linear(96, 4)

    def forward(self, s_samples, q_samples, s_labels, q_labels):
        features_labels = feature_extract(cfg=self.cfg, q_datas=q_samples, s_datas=s_samples,
                                          s_labels=s_labels, q_labels=q_labels, model=self.feature_extraction_model)
        support_features = features_labels['s_features']
        query_features = features_labels['q_features']
        cam_features = self.cross_attention(support_features, query_features)
        s_cam_f = cam_features['s_ca_f']
        q_cam_f = cam_features['q_ca_f']
        s_cam_f = self.avg2d1(s_cam_f)
        q_cam_f = self.avg2d2(q_cam_f)

        s_cam_f = s_cam_f.squeeze(2)
        q_cam_f = q_cam_f.squeeze(2)
        q_cam_f = F.relu(q_cam_f)
        q_cam_f = torch.flatten(q_cam_f, start_dim=1)

        binary_class = self.lin(q_cam_f)

        return binary_class


class CrossAttention_Distance_4_class(nn.Module):
    def __init__(self, cfg, device):
        super(CrossAttention_Distance_4_class, self).__init__()

        self.cross_attention = CrossAttention_feature_class(d_model=cfg.model_parameters.d_model,
                                                            n_head=cfg.model_parameters.num_heads,
                                                            dropout=cfg.model_parameters.dropout,
                                                            device=device)

        if cfg.model_parameters.model_name == 'LSTM':
            self.feature_extraction_model = get_LSTM_feature_extractor_model(cfg, device).requires_grad_(False)
        elif cfg.model_parameters.model_name == 'CNN':
            self.feature_extraction_model = get_cnn_feature_extractor_model(cfg, device).requires_grad_(False)
        else:
            self.feature_extraction_model = get_feature_extractor_model(cfg, device).requires_grad_(False)
        self.cfg = cfg
        self.device = device
        self.max2d1 = nn.MaxPool2d((5, 1))
        self.max2d2 = nn.MaxPool2d((5, 1))
        self.conv_block1 = conv_block()
        self.conv_block2 = conv_block()

        self.avg2d1 = nn.AvgPool2d((5, 1))
        self.avg2d2 = nn.AvgPool2d((5, 1))
        self.lin1 = nn.Linear(3, 4)
        self.lin2 = nn.Linear(3, 4)
        self.norm1 = nn.LayerNorm(cfg.model_parameters.d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(cfg.model_parameters.d_model, eps=1e-5)
        self.output_num = cfg.model_parameters.output_classes

    def forward(self, s_samples, q_samples, s_labels, q_labels):
        features_labels = feature_extract_4_class(cfg=self.cfg, q_datas=q_samples, s_datas=s_samples,
                                                  s_labels=s_labels, q_labels=q_labels,
                                                  model=self.feature_extraction_model)
        support_features = features_labels['s_features']
        query_features = features_labels['q_features']
        cam_features = self.cross_attention(support_features, query_features, self.output_num)
        s_cam_f = cam_features['s_ca_f']
        q_cam_f = cam_features['q_ca_f']
        # s_cam_f = self.max2d1(s_cam_f)
        # q_cam_f = self.max2d2(q_cam_f)
        s_cam_f = self.avg2d1(s_cam_f)
        q_cam_f = self.avg2d2(q_cam_f)

        # s_cam_f=s_cam_f.unsqueeze(1)
        # q_cam_f=q_cam_f.unsqueeze(1)

        # s_cam_f = self.conv_block1(s_cam_f)
        # q_cam_f = self.conv_block2(q_cam_f)
        #
        s_cam_f = s_cam_f.squeeze(2)
        q_cam_f = q_cam_f.squeeze(2)
        score = F.pairwise_distance(s_cam_f, q_cam_f, p=2)
        # score = torch.reshape(score, (self.cfg.model_parameters.classes,self.cfg.train_parameters.shot_num))
        # score = torch.mean(score, dim=-1)
        score = -score

        # support_features = self.max2d1(support_features)
        # query_features = self.max2d2(query_features)
        #
        # support_features = support_features.squeeze()
        # query_features = query_features.squeeze()
        # score = F.pairwise_distance(support_features, query_features, p=2)
        # score = -score

        return score


class CrossAttention_flatten_2class(nn.Module):
    def __init__(self, cfg, device):
        super(CrossAttention_flatten_2class, self).__init__()

        self.cross_attention = CrossAttention_feature_class(d_model=cfg.model_parameters.d_model,
                                                            n_head=cfg.model_parameters.num_heads,
                                                            dropout=cfg.model_parameters.dropout,
                                                            device=device)
        if cfg.model_parameters.model_name == 'LSTM':
            self.feature_extraction_model = get_LSTM_feature_extractor_model(cfg, device).requires_grad_(False)
        elif cfg.model_parameters.model_name == 'CNN':
            self.feature_extraction_model = get_cnn_feature_extractor_model(cfg, device).requires_grad_(False)
        else:
            self.feature_extraction_model = get_feature_extractor_model(cfg, device).requires_grad_(False)
        self.cfg = cfg
        self.device = device
        self.avg2d1 = nn.AvgPool2d((5, 1))
        self.avg2d2 = nn.AvgPool2d((5, 1))
        self.batch = nn.BatchNorm1d(num_features=cfg.model_parameters.d_model)
        self.lin = nn.Linear(96, 2)
        # self.lin = nn.Linear(64, 2)
        self.class_num = cfg.model_parameters.output_classes

    def forward(self, s_samples, q_samples, s_labels, q_labels):
        features_labels = feature_extract(cfg=self.cfg, q_datas=q_samples, s_datas=s_samples,
                                          s_labels=s_labels, q_labels=q_labels, model=self.feature_extraction_model)
        support_features = features_labels['s_features']
        query_features = features_labels['q_features']
        cam_features = self.cross_attention(support_features, query_features, self.class_num)
        s_cam_f = cam_features['s_ca_f']
        q_cam_f = cam_features['q_ca_f']
        s_cam_f = self.avg2d1(s_cam_f)
        q_cam_f = self.avg2d2(q_cam_f)

        s_cam_f = s_cam_f.squeeze(2)
        q_cam_f = q_cam_f.squeeze(2)
        # q_cam_f_relu = F.sigmoid(q_cam_f)
        q_cam_f_relu = F.relu(q_cam_f)
        q_cam_f_relu = torch.flatten(q_cam_f_relu, start_dim=1)

        binary_class = self.lin(q_cam_f_relu)

        return {'binary_class': binary_class, 's_cam_f': s_cam_f, 'q_cam_f': q_cam_f}



class CrossAttention_classifier(nn.Module):
    def __init__(self, cfg, device):
        super(CrossAttention_classifier, self).__init__()

        self.cross_attention = CrossAttention_feature_class(d_model=cfg.model_parameters.d_model,
                                                            n_head=cfg.model_parameters.num_heads,
                                                            dropout=cfg.model_parameters.dropout,
                                                            device=device)
        if cfg.model_parameters.model_name == 'LSTM':
            self.feature_extraction_model = get_LSTM_feature_extractor_model(cfg, device).requires_grad_(False)
        elif cfg.model_parameters.model_name == 'CNN':
            self.feature_extraction_model = get_cnn_feature_extractor_model(cfg, device).requires_grad_(False)
        else:
            self.feature_extraction_model = get_feature_extractor_model(cfg, device).requires_grad_(False)
        self.cfg = cfg
        self.device = device
        self.avg2d1 = nn.AvgPool2d((5, 1))
        self.avg2d2 = nn.AvgPool2d((5, 1))
        self.batch = nn.BatchNorm1d(num_features=cfg.model_parameters.d_model)
        self.lin = nn.Linear(96, 2)
        self.class_num = cfg.model_parameters.output_classes

    def forward(self, s_samples, q_samples, s_labels, q_labels):
        features_labels = feature_extract(cfg=self.cfg, q_datas=q_samples, s_datas=s_samples,
                                          s_labels=s_labels, q_labels=q_labels, model=self.feature_extraction_model)
        support_features = features_labels['s_features']
        query_features = features_labels['q_features']
        cam_features = self.cross_attention(support_features, query_features, self.class_num)
        s_cam_f = cam_features['s_ca_f']
        q_cam_f = cam_features['q_ca_f']
        s_cam_f = self.avg2d1(s_cam_f)
        q_cam_f = self.avg2d2(q_cam_f)

        s_cam_f = s_cam_f.squeeze(2)
        q_cam_f = q_cam_f.squeeze(2)
        q_cam_f_relu = F.sigmoid(q_cam_f)
        q_cam_f_relu = torch.flatten(q_cam_f_relu, start_dim=1)

        binary_class = self.lin(q_cam_f_relu)

        return {'binary_class': binary_class, 's_cam_f': s_cam_f, 'q_cam_f': q_cam_f}