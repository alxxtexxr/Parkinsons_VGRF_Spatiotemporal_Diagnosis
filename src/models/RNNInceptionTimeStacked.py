import torch
import torch.nn as nn

from tsai.models.InceptionTime import InceptionBlock
from tsai.models.layers import Permute, Concat, GAP1d

noop = nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class _RNNInceptionTimeStacked_Base(nn.Module):
    def __init__(self,
                 c_in, c_out, seq_len,
                 # RNN parameters
                 hidden_size=128, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout:float=0.,
                 # InceptionTime parameters
                 nf=32, 
                 ):
        super().__init__()
        
        # RNN layers
        self.rnn = self._cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0, 2, 1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        # InceptionTime layers
        self.inception_block = InceptionBlock(c_in, nf)
        inception_head_nf = nf * 4
        self.gap = GAP1d(1)
        
        # FC layers
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = nn.Linear(inception_head_nf, c_out)
    
    def forward(self, x):
        # RNN forward
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = self.rnn_dropout(rnn_output)
        
        # InceptionTime forward
        inception_block_out = self.inception_block(rnn_output)
        gap_out = self.gap(inception_block_out)
        
        # FC forward
        fc_out = self.fc(self.fc_dropout(gap_out))
        return fc_out

class RNNInceptionTimeStacked(_RNNInceptionTimeStacked_Base):
    _cell = nn.RNN
    
class GRUInceptionTime(_RNNInceptionTimeStacked_Base):
    _cell = nn.GRU

class LSTMInceptionTime(_RNNInceptionTimeStacked_Base):
    _cell = nn.LSTM

if __name__ == '__main__':
    n_feat = 16
    window_size = 500
    n_class = 4
    x = torch.randn(1, n_feat, window_size)
    model = RNNInceptionTimeStacked(c_in=n_feat, seq_len=window_size, c_out=n_class)
    output = model(x)
    print("Output shape:", output.shape)
