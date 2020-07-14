import torch
import torch.nn as nn
from torch.autograd import Variable

dtype = torch.FloatTensor


class TextLSTM(nn.Module):
    def __init__(self, vocaCount, n_hidden):
        super(TextLSTM, self).__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(input_size=vocaCount, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.rand(n_hidden, vocaCount).type(dtype))
        self.b = nn.Parameter(torch.rand(vocaCount).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)

        hidden_state = Variable(torch.zeros(1, len(X), self.n_hidden))
        cell_state = Variable(torch.zeros(1, len(X), self.n_hidden))

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]
        model = torch.mm(outputs, self.W) + self.b

        return model


