import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Task_04_Bi_LSTM import Trainer
from Task_04_Bi_LSTM.Bi_LSTM import BiLSTM

dtype = torch.FloatTensor

sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}

vocaSize = len(word_dict)
max_len = len(sentence.split())
n_hidden = 5

def make_batch(sentence):
    input_batch, target_batch = [], []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        target =word_dict[words[i + 1]]
        input_batch.append(np.eye(vocaSize)[input])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))


input_batch, target_batch = make_batch(sentence)

model = BiLSTM(vocaSize, n_hidden, dtype)

Trainer.train(model,input_batch, target_batch)

predict = model(input_batch).data.max(1, keepdim=True)[1]
print(sentence)
print([number_dict[n.item()] for n in predict.squeeze()])