import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear(self.hidden_size, hidden_size)
#         self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
#
#     def forward(self, lstm_outputs, final_hidden_state):
#         batch_size, seq_len, _ = lstm_outputs.shape
#         attn_weights = self.attn(lstm_outputs)
#         attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))
#         attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)
#         context = torch.bmm(lstm_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
#         attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))
#         return attn_hidden, attn_weights
#
#
# class LSTMClsssifier(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, num_layers, label_size):
#         super(LSTMClsssifier, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.label_size = label_size
#         self.lstmcal = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
#         self.hidden2label = nn.Linear(self.hidden_dim, label_size)
#         self.attn = Attention(hidden_dim)
#         self.hidden = None
#         self.rucal = nn.ReLU()
#         self.linears = nn.ModuleList()
#         self.linear_dims = [self.hidden_dim,5,3]
#         for i in range(0, 2):
#             self.linears.append(nn.Dropout(p=0.5))
#             linear_layer = nn.Linear(self.linear_dims[i], self.linear_dims[i + 1])
#             self.init_weights(linear_layer)
#             self.linears.append(linear_layer)
#             if i == len(self.linear_dims) - 1:
#                 break  # no activation after output layer!!!
#             self.linears.append(nn.ReLU())
#
#     def init_weights(self, layer):
#         if type(layer) == nn.Linear:
#             # print("Initialize layer with nn.init.xavier_uniform_: {}".format(layer))
#             torch.nn.init.xavier_uniform_(layer.weight)
#             layer.bias.data.fill_(0.01)
#
#
#     def init_hidden(self, batch_size):
#         return (torch.zeros((self.num_layers,batch_size, self.hidden_dim)),
#                 torch.zeros((self.num_layers,batch_size,  self.hidden_dim)))
#
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         h0,c0 = self.init_hidden(batch_size)
#         lstm_output, self.hidden = self.lstmcal(x,(h0,c0))
#         # final_state = self.hidden[0].view(self.num_layers, 1, batch_size, self.hidden_dim)[-1]
#         # final_hidden_state = final_state.squeeze(0)
#         # x, att_weights = self.attn(lstm_output, final_hidden_state)
#         #
#         # for l in self.linears:
#         #     x = l(x)
#         # x = self.hidden2label(x)
#         # x = self.rucal(x)
#         x = self.hidden2label(x)
#
#
#         x = F.log_softmax(x, dim=1)
#
#
#         return x


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

        # self.lstm1a = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=1,
        #                       dropout=0, batch_first=True)
        # self.lstm1b = nn.LSTM(input_size=hidden_dim, hidden_size=self.hidden_dim // 2, num_layers=1,
        #                       dropout=0, batch_first=True)
        # self.lstm1c = nn.LSTM(input_size=hidden_dim//2, hidden_size=self.hidden_dim // 4, num_layers=1,
        #                       dropout=0, batch_first=True)
        #
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

#
# class LSTMClsssifier(nn.Module):
#     def __init__(self, input_size, hidden_dim, num_layers, label_size, seq_length):
#         super(LSTMClsssifier, self).__init__()
#         self.input_size = input_size
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.label_size = label_size
#         self.seq_length = seq_length
#         self.lstmcal = nn.LSTM(input_size=input_size,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True)
#         self.fc = nn.Linear(hidden_dim,label_size)
#
#     def forward(self,x):
#         batch,_,_=x.shape
#         h_0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))
#         c_0 = Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))
#         ula,(h_out,_) = self.lstmcal(x,(h_0,c_0))
#         h_out = h_out.view(-1,self.hidden_dim)
#         out = self.fc(h_out)
#         return out

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