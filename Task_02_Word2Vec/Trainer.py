import Utills
from Task_02_Word2Vec.Word2Vec import Word2Vec
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch


def train(maxEpoch, skip_grams, batch_size, voc_size, embedding_size, dtype):

    model = Word2Vec(voc_size, embedding_size, dtype)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(maxEpoch):
        input_batch, target_batch = Utills.random_batch(skip_grams, batch_size, voc_size)

        input_batch = Variable(torch.Tensor(input_batch))
        target_batch = Variable(torch.LongTensor(target_batch))

        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    return model