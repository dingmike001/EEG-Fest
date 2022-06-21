import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionTransformer_Ning(nn.Module):
    def __init__(self, eeg_channel, d_model, n_head, d_hid, n_layers, dropout=0.5,
                 seq_len=1600, device='cpu', use_model_selection=True):
        super(AttentionTransformer_Ning, self).__init__()
        self.model_type = 'Transformer'
        self.use_model_selection = use_model_selection
        max_len = int((seq_len - 1) // 2)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_hid,
                                                    dropout=dropout, batch_first=True, norm_first=False, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model

        self.dropf = nn.Dropout(p=dropout)
        self.linear_in = nn.Linear(eeg_channel, d_model)

    def forward(self, src):

        src = self.linear_in(src)
        src = self.dropf(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 32, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()

        self.dropf = nn.Dropout(p=dropout)

        flag = (d_model % 2 == 0)
        if flag:
            d_model_new = d_model
        else:
            d_model_new = d_model + 1

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_new, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model_new)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        if not flag:
            pe = pe[:, :, d_model - 1]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len,  embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropf(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, classes, dropout, d_model):
        super().__init__()
        self.fla = nn.Flatten(start_dim=1)
        self.dropf = nn.Dropout(p=dropout)
        self.ln1 = nn.Linear(d_model, 8)
        nn.init.xavier_normal_(self.ln1.weight)
        self.ln2 = nn.Linear(input_dim, classes)
        nn.init.xavier_normal_(self.ln2.weight)

    def forward(self, x):
        x = F.sigmoid(self.ln1(x))
        x = self.dropf(x)
        x = self.fla(x)
        x = self.ln2(x)
        return x
