import torch.optim as optim
import torch.nn as nn

def train(model, input, output, maxEpoch):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    for epoch in range(maxEpoch):
        optimizer.zero_grad()
        answer = model(input)

        loss = criterion(answer, output)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()