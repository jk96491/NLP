import torch
import torch.nn as nn

class NNLM_Net(nn.Module):
    def __init__(self, vocaCount, n_gram_size, m, n_hidden, dtype):
        super(NNLM_Net, self).__init__()
        self.concatSize = n_gram_size * m

        self.EmbbeddingLayer = nn.Embedding(vocaCount, m)

        self.HiddenLayer = nn.Parameter(torch.randn(n_gram_size * m, n_hidden).type(dtype))
        self.HiddenLayerBias = nn.Parameter(torch.randn(n_hidden).type(dtype))

        self.OutputLayer = nn.Parameter(torch.randn(n_hidden, vocaCount).type(dtype))
        self.OutputLayerBias = nn.Parameter(torch.randn(vocaCount).type(dtype))

    def forward(self, input):
        X = self.EmbbeddingLayer(input)
        X = X.view(-1, self.concatSize)

        tanh = torch.tanh(self.HiddenLayerBias + torch.mm(X, self.HiddenLayer))
        output = self.OutputLayerBias + torch.mm(tanh, self.OutputLayer)

        return output
