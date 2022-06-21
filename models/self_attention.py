import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 32, dropout: float = 0.1, max_len: int = 1000, device: str = 'cpu'):
        super().__init__()

        self.dropf = nn.Dropout(p=dropout)

        flag = (d_model % 2 == 0)
        if flag:
            d_model_new = d_model
        else:
            d_model_new = d_model + 1

        position = torch.arange(max_len).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model_new, 2) * (-math.log(10000.0) / d_model)).to(device)
        pe = torch.zeros(1, max_len, d_model_new).to(device)
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

def Convolution_Layer(in_num, out_num, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_num),
        nn.ReLU())



class AttentiontTransformer(nn.Module):
    def __init__(self, eeg_channel: int, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.5,
                 max_len: int = 799, device='cpu'):
        super(AttentiontTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len, device=device)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_hid,
                                                    dropout=dropout, batch_first=True, norm_first=False, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model
        self.lin1 = nn.Linear(d_model, 8)
        nn.init.xavier_normal_(self.lin1.weight)
        self.lin2 = nn.Linear(6392, 3)
        nn.init.xavier_normal_(self.lin2.weight)
        self.sftmax = nn.Softmax(dim=1)
        self.conv = Convolution_Layer(eeg_channel, 32, 5, 1, 0)
        self.dropf = nn.Dropout(p=dropout)
        self.liniar_in = nn.Linear(d_model, d_model)
        self.fla = nn.Flatten(start_dim=1)

    def decoder(self, x):
        x = F.sigmoid(self.lin1(x))
        x = self.dropf(x)
        x = self.fla(x)

        x = self.lin2(x)
        x = self.sftmax(x)
        return x

    def forward(self, src):
        src = self.conv(src)
        src = self.dropf(src)
        src = src.squeeze(-1)
        src = torch.transpose(src, 2, 1)
        src = self.liniar_in(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output