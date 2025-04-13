import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math



class LSTMClsssifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, label_size, seq_length):
        super(LSTMClsssifier, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.label_size = label_size
        self.seq_length = seq_length
        self.lstmcal = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                               dropout=0, batch_first=True)
        self.lstmcala = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim // 2,
                                num_layers=self.num_layers,
                                dropout=0, batch_first=True)
        self.lstmcalaa = nn.LSTM(input_size=self.hidden_dim // 2, hidden_size=self.hidden_dim // 4,
                                 num_layers=self.num_layers,
                                 dropout=0, batch_first=True)
        self.linear4th = nn.Linear(self.hidden_dim * self.seq_length // 4, self.label_size)

        self.l_linear1 = torch.nn.Linear(self.hidden_dim // 5 * self.seq_length, 50)
        self.l_lineardeter = torch.nn.Linear(self.hidden_dim * self.seq_length, self.label_size)
        self.l_linear2 = nn.Linear(50, self.label_size)
        self.leakrul = nn.LeakyReLU(negative_slope=0.02)
        self.sof = nn.Softmax(dim=1)
        self.hidden2label = nn.Linear(self.hidden_dim, label_size)
        self.l_linear3 = nn.Linear(51, self.label_size)
        self.linear1aa = nn.Linear(425, label_size)

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)
        return hidden

    def init_hiddena(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 2)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 2)
        hidden = (hidden_state, cell_state)
        return hidden

    def init_hiddenaa(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 4)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 4)
        hidden = (hidden_state, cell_state)
        return hidden

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hidden = self.init_hidden(batch_size)
        x, hidden_out = self.lstmcal(x, hidden)
        hidden = self.init_hiddena(batch_size)
        x, hidden_out = self.lstmcala(x, hidden)
        hidden = self.init_hiddenaa(batch_size)
        x, hidden_out = self.lstmcalaa(x, hidden)
        x = x.reshape(batch_size, -1)
        x = self.linear4th(x)

        return x



class AttentionTransformer_Ning(nn.Module):
    def __init__(self, eeg_channel=17, d_model=32, n_head=8, d_hid=2048, n_layers=3, dropout=0.1,
                 seq_len=1600, device='cuda'):
        super(AttentionTransformer_Ning, self).__init__()
        self.model_type = 'Transformer'
        max_len = int((seq_len - 1) // 2)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_hid,
                                                    dropout=dropout, batch_first=True, norm_first=False, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model

        self.dropf = nn.Dropout(p=dropout)
        self.linear_in = nn.Linear(eeg_channel, d_model)
        self.fc1 = nn.Linear(d_model, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.avg2d1 = nn.AvgPool2d((5, 1))

    def forward(self, src):
        # src = torch.transpose(src,2,1)
        src = self.linear_in(src)
        src = self.dropf(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.avg2d1(output)
        output = output.squeeze(1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)


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



class LSTM(nn.Module):
    """Long Short Term Memory"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        if self.bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class AttentionalLSTM(nn.Module):
    """LSTM with Attention"""
    def __init__(self, input_size, qkv, hidden_size, num_layers, output_size, bidirectional=False):
        super(AttentionalLSTM, self).__init__()

        self.input_size = input_size
        self.qkv = qkv
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.query = nn.Linear(input_size, qkv)
        self.key = nn.Linear(input_size, qkv)
        self.value = nn.Linear(input_size, qkv)

        self.attn = nn.Linear(qkv, input_size)
        self.scale = math.sqrt(qkv)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        Q, K, V = self.query(x), self.key(x), self.value(x)

        dot_product = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
        scores = torch.softmax(dot_product, dim=-1)
        scaled_x = torch.matmul(scores, V) + x

        out = self.attn(scaled_x) + x
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = self.fc(out)

        return out
