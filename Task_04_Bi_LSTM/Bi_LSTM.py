import torch
import torch.nn as nn
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, vocaSize, n_hidden, dtype):
        super(BiLSTM, self).__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(input_size=vocaSize, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Parameter(torch.rand(n_hidden * 2, vocaSize).type(dtype))
        self.b = nn.Parameter(torch.rand(vocaSize).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)

        hidden_state = Variable(torch.zeros(1*2, len(X), self.n_hidden))
        cell_state = Variable(torch.zeros(1*2, len(X), self.n_hidden))

        output, (_, _) = self.lstm(input, (hidden_state,  cell_state))
        output = output[-1]
        model = torch.mm(output, self.W) + self.b

        return model