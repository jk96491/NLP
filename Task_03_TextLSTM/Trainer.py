import torch.nn as nn
import torch.optim as optim
from Task_03_TextLSTM.TextLSTM import TextLSTM


def train(maxEpoch, input_batch, target_batch, vocaCount, n_hidden):
    model = TextLSTM(vocaCount, n_hidden)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    output = model(input_batch)

    for epoch in range(maxEpoch):
        optimizer.zero_grad()

        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    return model