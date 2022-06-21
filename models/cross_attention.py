import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        pe = pe.to('cuda')
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len,  embedding_dim]
        """

        x = x + self.pe[:, :x.size(1), :]
        return self.dropf(x)


class TransformerCAMEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5):
        super(TransformerCAMEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, s, q):
        x = self.norm1(q + self._ca_block(s, q))
        x = self.norm2(q + self._ff_block(x))
        return x

    def _ca_block(self, s, q):
        x = self.attention(q, s, s)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)


class CrossAttention_feature(nn.Module):
    def __init__(self, d_model: int = 32, n_head: int = 8, dropout: float = 0.1, device='cpu'):
        super(CrossAttention_feature, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.pos_encoder1 = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.pos_encoder2 = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.C_A_M1 = TransformerCAMEncoderLayer(d_model=self.d_model, nhead=self.n_head, dropout=dropout).to(device)
        self.C_A_M2 = TransformerCAMEncoderLayer(d_model=self.d_model, nhead=self.n_head, dropout=dropout).to(device)
        # self.transformer_encoder1 = nn.TransformerEncoder(C_A_M1, num_layers=2)
        # self.transformer_encoder2 = nn.TransformerEncoder(C_A_M2, num_layers=2)

    def datarestack(self, xs):
        num = xs.shape[0]
        for i in range(num):
            x = xs[i]
            if i == 0:
                output = x
            else:
                output = torch.cat((output, x), dim=0)
        return output

    def dataunstack(self, xs, batch_size, class_num):
        for b in range(batch_size):
            start = b * class_num + 0
            end = b * class_num + class_num
            x = xs[start:end, :, :]
            if b == 0:
                output = x.unsqueeze(0)
            else:
                output = torch.cat((output, x.unsqueeze(0)), dim=0)
        return (output)

    def forward(self, s, q):
        batch_size = s.shape[0]
        class_num = 3
        s = self.datarestack(s)
        q = self.datarestack(q)
        s = self.pos_encoder1(s)
        q = self.pos_encoder2(q)
        qq = self.C_A_M1(s, q)
        ss = self.C_A_M2(q, s)
        qq = self.dataunstack(qq, batch_size, class_num)
        ss = self.dataunstack(ss, batch_size, class_num)

        return {'s_ca_f': ss, 'q_ca_f': qq}


class CrossAttention_feature_class(nn.Module):
    def __init__(self, d_model: int = 32, n_head: int = 8, dropout: float = 0.1, device='cpu'):
        super(CrossAttention_feature_class, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.pos_encoder1 = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.pos_encoder2 = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.C_A_M1 = TransformerCAMEncoderLayer(d_model=self.d_model, nhead=self.n_head, dropout=dropout).to(device)
        self.C_A_M2 = TransformerCAMEncoderLayer(d_model=self.d_model, nhead=self.n_head, dropout=dropout).to(device)
        # self.transformer_encoder1 = nn.TransformerEncoder(C_A_M1, num_layers=2)
        # self.transformer_encoder2 = nn.TransformerEncoder(C_A_M2, num_layers=2)

    def datarestack(self, xs):
        num = xs.shape[0]
        for i in range(num):
            x = xs[i]
            if i == 0:
                output = x
            else:
                output = torch.cat((output, x), dim=0)
        return output

    def dataunstack(self, xs, batch_size, class_num):
        for b in range(batch_size):
            start = b * class_num + 0
            end = b * class_num + class_num
            x = xs[start:end, :, :]
            if b == 0:
                output = x.unsqueeze(0)
            else:
                output = torch.cat((output, x.unsqueeze(0)), dim=0)
        return (output)

    def forward(self, s, q,n):
        batch_size = s.shape[0]
        class_num = n
        s = self.datarestack(s)
        q = self.datarestack(q)
        s = self.pos_encoder1(s)
        q = self.pos_encoder2(q)
        qq = self.C_A_M1(s, q)
        ss = self.C_A_M2(q, s)
        qq = self.dataunstack(qq, batch_size, class_num)
        ss = self.dataunstack(ss, batch_size, class_num)

        return {'s_ca_f': ss, 'q_ca_f': qq}
