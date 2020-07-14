import torch
from Task_04_Bi_LSTM import Trainer
from Task_04_Bi_LSTM.Bi_LSTM import BiLSTM
import Utills

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

input_batch, target_batch = Utills.make_batchLongSentence(sentence, word_dict, max_len, vocaSize)

model = BiLSTM(vocaSize, n_hidden, dtype)

Trainer.train(model,input_batch, target_batch)

predict = model(input_batch).data.max(1, keepdim=True)[1]
print(sentence)
print([number_dict[n.item()] for n in predict.squeeze()])