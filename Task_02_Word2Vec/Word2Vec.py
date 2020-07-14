import torch
import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self, voc_size, embedding_size, dtype):
        super(Word2Vec, self).__init__()

        self.W = nn.Parameter(-2 * torch.rand(voc_size, embedding_size) + 1).type(dtype)
        self.WT = nn.Parameter(-1 * torch.rand(embedding_size, voc_size) + 1).type(dtype)

    def forward(self, X):
        hiddenLayer = torch.mm(X, self.W)
        outputLayer = torch.mm(hiddenLayer, self.WT)

        return outputLayer











